"""单用户本地知识库版本。没有网络调用；向量化仅发生在创建新版本时。

每个版本独立Chroma目录，manifest.json最后原子写入作为提交标记。
失败的未提交目录保留供排错，不覆盖、删除任何旧版本。
"""

import hashlib
import json
import os
import re
from datetime import datetime, timezone
from importlib.metadata import PackageNotFoundError, version
from pathlib import Path
from time import perf_counter
from uuid import uuid4

SCHEMA_VERSION = 1
DATA_ROOT = Path(__file__).resolve().parent / "_local_kb"


def sha256(data):
    return hashlib.sha256(data).hexdigest()


def atomic_json(path, value):
    path = Path(path)
    temporary = path.with_name(path.name + "." + uuid4().hex + ".tmp")
    try:
        with temporary.open("x", encoding="utf-8", newline="\n") as handle:
            json.dump(value, handle, ensure_ascii=False, indent=2, allow_nan=False)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)


def embedding_identity(config):
    # 不保存密钥或接口原文。更换密钥/聊天模型不改变已有向量空间。
    # 接口仅规范化尾斜线；即使相同模型名，跨供应商也拒绝直接混用。
    endpoint = config["SILICONFLOW_BASE_URL"].strip().rstrip("/")
    return {
        "model": config["EMBEDDING_MODEL"].strip(),
        "provider_sha256": sha256(endpoint.encode("utf-8")),
        "options": {"check_embedding_ctx_length": False, "encoding_format": "float"},
    }


def runtime_versions():
    result = {}
    for name in ("chromadb", "langchain-chroma", "langchain-openai", "langchain-core"):
        try:
            result[name] = version(name)
        except PackageNotFoundError:
            result[name] = "not-installed"
    return result


def version_directory(root, kb_id, version_id):
    if not re.fullmatch(r"kb_[a-f0-9]{32}", kb_id or ""):
        raise ValueError("知识库标识无效。")
    if not re.fullmatch(r"v_[a-f0-9]{32}", version_id or ""):
        raise ValueError("知识库版本标识无效。")
    return Path(root) / kb_id / "versions" / version_id


def load_manifest(root, kb_id, version_id):
    path = version_directory(root, kb_id, version_id) / "manifest.json"
    manifest = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(manifest, dict):
        raise ValueError("知识库版本信息格式不正确。")
    if (manifest.get("schema_version") != SCHEMA_VERSION
            or manifest.get("kb_id") != kb_id or manifest.get("version_id") != version_id
            or manifest.get("collection_name") != "product_" + version_id[2:]):
        raise ValueError("知识库版本信息不兼容或损坏；请保留目录并创建新版本。")
    required = ("name", "created_at_utc", "embedding_identity", "runtime_versions",
                "files", "chunk_count", "pdf_pages", "parse_warnings", "chunks_sha256",
                "build_snapshot")
    if any(key not in manifest for key in required):
        raise ValueError("知识库版本信息不完整。")
    if (not isinstance(manifest["created_at_utc"], str)
            or not isinstance(manifest["name"], str)
            or not isinstance(manifest["files"], list)
            or not isinstance(manifest["chunk_count"], int)
            or not isinstance(manifest["runtime_versions"], dict)
            or not isinstance(manifest["embedding_identity"], dict)
            or not isinstance(manifest["build_snapshot"], dict)
            or "run_id" not in manifest["build_snapshot"]):
        raise ValueError("知识库版本信息类型不正确。")
    for item in manifest["files"]:
        if (not isinstance(item, dict)
                or any(key not in item for key in ("name", "bytes", "sha256", "chunk_count"))
                or not isinstance(item["name"], str)):
            raise ValueError("知识库资料清单格式不正确。")
    return manifest


def list_versions(root=DATA_ROOT):
    entries, warnings = [], []
    if not Path(root).exists():
        return entries, warnings
    for directory in sorted(Path(root).glob("kb_*/versions/v_*")):
        if not (directory / "manifest.json").is_file():
            warnings.append(f"未完成的版本已忽略：{directory.parent.parent.name}/{directory.name}")
            continue
        try:
            entries.append(load_manifest(root, directory.parent.parent.name, directory.name))
        except (ValueError, KeyError, OSError, TypeError):
            warnings.append(f"无法读取的版本已忽略：{directory.parent.parent.name}/{directory.name}")
    entries.sort(key=lambda item: item["created_at_utc"], reverse=True)
    return entries, warnings


def validate_files(files):
    if not 0 < len(files) <= 5:
        raise ValueError("每个版本请上传1至5份完整资料。")
    names = [name.replace("\\", "/").rsplit("/", 1)[-1] for name, _ in files]
    if len({name.casefold() for name in names}) != len(names):
        raise ValueError("存在同名资料，请先改名，避免同名文件的检索片段混在一起。")
    if any(not data for _, data in files):
        raise ValueError("存在空文件，请检查后重新上传。")
    return [(name, data) for name, (_, data) in zip(names, files)]


def validate_store(store, chunks):
    """核对持久索引与块清单；get不调用嵌入接口，也不修改集合。"""
    result = store.get(include=["documents", "metadatas"])
    actual = dict(zip(result["ids"], zip(result["documents"], result["metadatas"])))
    expected = {f"chunk_{index}": (doc.page_content, doc.metadata)
                for index, doc in enumerate(chunks)}
    if actual != expected:
        raise ValueError("保存的向量索引与文本块不一致，请保留旧版本并重新建库。")


def create_version(files, config, name, parse, indexes, embeddings_factory,
                   chain_factory, snapshot_factory, root=DATA_ROOT, kb_id=None):
    name = name.strip()
    if not name or len(name) > 60:
        raise ValueError("请填写1至60字的知识库名称。")
    files = validate_files(files)
    if kb_id:
        # 更新必须指向已经发布的知识库，名称沿用，不能任意接入目录。
        prior, _ = list_versions(root)
        prior = [item for item in prior if item["kb_id"] == kb_id]
        if not prior:
            raise ValueError("未找到要更新的知识库，请先打开已保存版本。")
        name = prior[0]["name"]
    kb_id = kb_id or "kb_" + uuid4().hex
    version_id = "v_" + uuid4().hex
    directory = version_directory(root, kb_id, version_id)
    started = perf_counter()
    chunks, warnings, pages = parse(files)
    keywords, lookup, neighbors = indexes(chunks)
    # 创建模型客户端不生成答案；尽早发现配置/依赖问题。
    chain = chain_factory(config)
    embeddings = embeddings_factory(config)
    directory.mkdir(parents=True, exist_ok=False)
    from langchain_chroma import Chroma
    from chromadb.config import Settings
    store = Chroma(collection_name="product_" + version_id[2:],
                   embedding_function=embeddings, persist_directory=str(directory / "chroma"),
                   client_settings=Settings(anonymized_telemetry=False))
    # 永不在异常处理或切库时delete_collection：旧版本及未提交版本均不覆盖。
    store.add_documents(chunks, ids=[f"chunk_{i}" for i in range(len(chunks))])
    validate_store(store, chunks)
    (directory / "sources").mkdir()
    file_info = []
    for i, (filename, data) in enumerate(files):
        stored = f"sources/{i:03d}{Path(filename).suffix.lower()}"
        (directory / stored).write_bytes(data)
        file_info.append({"name": filename, "bytes": len(data), "sha256": sha256(data),
                          "stored": stored,
                          "chunk_count": sum(d.metadata["source"] == filename for d in chunks)})
    atomic_json(directory / "chunks.json", [
        {"page_content": doc.page_content, "metadata": doc.metadata} for doc in chunks])
    manifest = {
        "schema_version": SCHEMA_VERSION, "kb_id": kb_id, "version_id": version_id,
        "name": name, "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "collection_name": "product_" + version_id[2:],
        "embedding_identity": embedding_identity(config),
        "runtime_versions": runtime_versions(), "files": file_info,
        "chunk_count": len(chunks), "pdf_pages": pages, "parse_warnings": warnings,
        "chunks_sha256": sha256((directory / "chunks.json").read_bytes()),
    }
    kb = {"store": store, "chain": chain, "keyword_by_source": keywords,
          "chunk_lookup": lookup, "neighbors": neighbors, "chunks": len(chunks),
          "files": len(files), "pages": pages, "warnings": warnings,
          "names": [filename for filename, _ in files], "manifest": manifest}
    snapshot = snapshot_factory(kb, files, config, perf_counter() - started)
    snapshot.update(knowledge_base_id=kb_id, knowledge_base_name=name,
                    knowledge_base_version=version_id, index_action="created")
    manifest["build_snapshot"] = snapshot
    # 提交标记最后写；失败不影响任何之前可用版本。
    atomic_json(directory / "manifest.json", manifest)
    kb["snapshot"] = dict(snapshot)
    return kb


def open_version(kb_id, version_id, config, indexes, embeddings_factory,
                 chain_factory, root=DATA_ROOT):
    started = perf_counter()
    manifest = load_manifest(root, kb_id, version_id)
    if manifest["embedding_identity"] != embedding_identity(config):
        raise ValueError("嵌入模型或服务接口与建库时不同，已阻止加载旧向量。"
                         "请恢复原配置后重新打开，或用完整资料创建新版本。")
    # Chroma升级可能迁移磁盘格式；保守阻止不一致环境打开旧目录。
    if manifest["runtime_versions"].get("chromadb") != runtime_versions()["chromadb"]:
        raise ValueError("chromadb版本与建库时不同。请恢复原依赖，或备份旧目录后用资料建立新版本。")
    directory = version_directory(root, kb_id, version_id)
    raw = (directory / "chunks.json").read_bytes()
    if sha256(raw) != manifest["chunks_sha256"]:
        raise ValueError("文本块文件校验失败，请保留旧版本并重新建库。")
    from langchain_core.documents import Document
    chunks = [Document(page_content=item["page_content"], metadata=item["metadata"])
              for item in json.loads(raw)]
    if len(chunks) != manifest["chunk_count"] or not chunks:
        raise ValueError("知识库文本块数量不一致。")
    for i, item in enumerate(manifest["files"]):
        # 固定位置读取，禁止元数据中的路径改变读取范围。
        suffix = Path(item["name"]).suffix.lower()
        if suffix not in (".pdf", ".txt"):
            raise ValueError("资料类型异常。")
        data = (directory / "sources" / f"{i:03d}{suffix}").read_bytes()
        if len(data) != item["bytes"] or sha256(data) != item["sha256"]:
            raise ValueError("保存的原始资料校验失败，请保留旧版本并重新建库。")
    if not (directory / "chroma" / "chroma.sqlite3").is_file():
        raise ValueError("向量索引缺失，请用完整资料创建新版本。")
    from langchain_chroma import Chroma
    from chromadb.config import Settings
    # 禁止在打开失败时悄悄创建空集合；此路径没有add_documents/embed_documents。
    store = Chroma(collection_name=manifest["collection_name"],
                   embedding_function=embeddings_factory(config),
                   persist_directory=str(directory / "chroma"),
                   client_settings=Settings(anonymized_telemetry=False),
                   create_collection_if_not_exists=False)
    validate_store(store, chunks)
    keywords, lookup, neighbors = indexes(chunks)
    snapshot = dict(manifest["build_snapshot"])
    snapshot.update(run_id=uuid4().hex, index_action="reopened",
                    original_build_run_id=manifest["build_snapshot"]["run_id"],
                    open_seconds=round(perf_counter() - started, 4))
    return {"store": store, "chain": chain_factory(config),
            "keyword_by_source": keywords, "chunk_lookup": lookup, "neighbors": neighbors,
            "chunks": len(chunks), "files": len(manifest["files"]),
            "pages": manifest["pdf_pages"], "warnings": manifest["parse_warnings"],
            "names": [item["name"] for item in manifest["files"]],
            "manifest": manifest, "snapshot": snapshot}
