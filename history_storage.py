"""单用户问答归档。仅Python标准库；不读取.env、不调用模型、不修改向量库。"""
from contextlib import contextmanager
import copy
from datetime import datetime, timezone
import json
from pathlib import Path
import sqlite3

SCHEMA_VERSION = 1
RECORD_KEYS = ('record_id', 'created_at_utc', 'case_id', 'status', 'question', 'answer',
               'sources', 'notes', 'telemetry', 'citation_audit', 'kb_snapshot',
               'manual_review', 'error_type')
SNAPSHOT_KEYS = set(('run_id app_version app_sha256 evaluation_tools_sha256 '
    'kb_storage_sha256 history_storage_sha256 runtime_versions prompt_version '
    'retrieval_version chat_model embedding_model chunk_size chunk_overlap '
    'vector_candidates seeds_per_file max_context_chars uploaded_files active_sources '
    'chunk_count pdf_pages parse_warnings build_seconds knowledge_base_id '
    'knowledge_base_name knowledge_base_version index_action original_build_run_id '
    'open_seconds followup_version followup_tools_sha256 '
    'answer_checks_version answer_checks_sha256').split())
SAFE_SNAPSHOT_KEYS = SNAPSHOT_KEYS | {'build_' + k for k in SNAPSHOT_KEYS}


class HistoryError(Exception):
    pass


class HistoryConflict(HistoryError):
    pass


def now():
    return datetime.now(timezone.utc).isoformat()


def clean_record(record):
    """仅接受评测记录白名单；不保存客户端对象、配置、密钥或内部签名。"""
    if not isinstance(record, dict) or any(k not in record for k in RECORD_KEYS):
        raise HistoryError('问答记录字段不完整，未保存。')
    result = copy.deepcopy({k: record[k] for k in RECORD_KEYS})
    if 'question_resolution' in record:
        value = record['question_resolution']
        if not isinstance(value, dict):
            raise HistoryError('问题补全追溯格式不正确，未保存。')
        result['question_resolution'] = copy.deepcopy({k: v for k, v in value.items() if k in (
            'version', 'status', 'original_question', 'retrieval_question', 'used_record_ids',
            'considered_record_id', 'conversation_id', 'knowledge_base_id', 'knowledge_base_version',
            'rule', 'reason', 'clarification')})
    if 'retrieval_question' in record:
        result['retrieval_question'] = record['retrieval_question']
    if (not isinstance(result['record_id'], str) or not result['record_id']
            or result['status'] not in ('success', 'failed')
            or not isinstance(result['question'], str)
            or not isinstance(result['answer'], str)
            or not isinstance(result['sources'], list)
            or any(not isinstance(s, dict) for s in result['sources'])
            or not isinstance(result['telemetry'], dict)
            or not isinstance(result['citation_audit'], dict)
            or not isinstance(result['kb_snapshot'], dict)
            or not isinstance(result['manual_review'], dict)):
        raise HistoryError('问答记录格式不正确，未保存。')
    snapshot = {k: v for k, v in result['kb_snapshot'].items() if k in SAFE_SNAPSHOT_KEYS}
    for key in ('runtime_versions', 'build_runtime_versions'):
        if key in snapshot and isinstance(snapshot[key], dict):
            snapshot[key] = {k: v for k, v in snapshot[key].items()
                             if k in ('chromadb', 'langchain-chroma', 'langchain-openai', 'langchain-core')}
    for key in ('uploaded_files', 'build_uploaded_files'):
        if key in snapshot:
            if not isinstance(snapshot[key], list) or any(not isinstance(i, dict) for i in snapshot[key]):
                raise HistoryError('资料快照格式不正确，未保存。')
            snapshot[key] = [{k: item[k] for k in ('name', 'bytes', 'sha256') if k in item}
                             for item in snapshot[key]]
    result['kb_snapshot'] = snapshot
    result['telemetry'] = {k: v for k, v in result['telemetry'].items() if k in (
        'model_calls', 'citation_retry_count', 'retrieval_seconds', 'retrieved_count',
        'generation_seconds', 'total_seconds', 'evidence_retry_count', 'answer_checks')}
    if 'answer_checks' in result['telemetry']:
        stages = result['telemetry']['answer_checks']
        if not isinstance(stages, list) or any(not isinstance(s, dict) for s in stages):
            raise HistoryError('数值检查追溯格式不正确，未保存。')
        cleaned = []
        for stage in stages:
            stage = {k: v for k, v in stage.items() if k in (
                'stage', 'answer', 'check', 'error_type', 'retry_question', 'decision')}
            if 'decision' in stage:
                if not isinstance(stage['decision'], dict):
                    raise HistoryError('校正采纳记录格式不正确，未保存。')
                stage['decision'] = {k: v for k, v in stage['decision'].items() if k in (
                    'accepted', 'reason', 'before_issue_count', 'after_issue_count', 'removed_supported_quantities')}
            if 'check' in stage:
                check = stage['check']
                if not isinstance(check, dict) or not isinstance(check.get('issues', []), list):
                    raise HistoryError('数值检查结果格式不正确，未保存。')
                check = {k: v for k, v in check.items() if k in (
                    'version', 'checked_quantities', 'issues', 'status', 'limitation')}
                check['issues'] = [{k: v for k, v in issue.items() if k in (
                    'code', 'source_id', 'message', 'quantity', 'qualifiers')}
                    for issue in check.get('issues', []) if isinstance(issue, dict)]
                stage['check'] = check
            cleaned.append(stage)
        result['telemetry']['answer_checks'] = cleaned
    result['citation_audit'] = {k: v for k, v in result['citation_audit'].items()
                                if k in ('valid_ids', 'invalid_ids', 'has_numeric_citation')}
    result['sources'] = [{k: item[k] for k in ('id', 'label', 'text') if k in item}
                         for item in result['sources']]
    result['manual_review'] = {k: result['manual_review'].get(k, '' if k == 'comment' else '待核验')
                               for k in ('fact', 'citation_support', 'retrieval_coverage', 'comment')}
    # 此处只保证可序列化，不修改用户输入或资料正文。
    try:
        return json.loads(json.dumps(result, ensure_ascii=False, allow_nan=False))
    except (TypeError, ValueError) as exc:
        raise HistoryError('问答记录不能序列化，未保存。') from exc


def encode(record):
    return json.dumps(record, ensure_ascii=False, sort_keys=True, allow_nan=False)


@contextmanager
def connection(path):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    con = sqlite3.connect(str(path), timeout=2)
    try:
        con.execute('PRAGMA foreign_keys = ON')
        version = con.execute('PRAGMA user_version').fetchone()[0]
        if version not in (0, SCHEMA_VERSION):
            raise HistoryError('历史数据库版本不兼容；请保留文件，不要覆盖。')
        if version == 0:
            tables = con.execute("SELECT name FROM sqlite_master WHERE type='table'").fetchall()
            if tables:
                raise HistoryError('历史数据库包含未知结构；请保留文件，不要覆盖。')
            con.execute('BEGIN IMMEDIATE')
            con.execute('CREATE TABLE records (seq INTEGER PRIMARY KEY AUTOINCREMENT, '
                        'record_id TEXT UNIQUE NOT NULL, revision INTEGER NOT NULL, body TEXT NOT NULL)')
            con.execute('CREATE TABLE conversations (conversation_id TEXT PRIMARY KEY, '
                        'kb_id TEXT NOT NULL, version_id TEXT NOT NULL, title TEXT NOT NULL, '
                        'created_at_utc TEXT NOT NULL, updated_at_utc TEXT NOT NULL)')
            con.execute('CREATE TABLE conversation_records (conversation_id TEXT NOT NULL '
                        'REFERENCES conversations(conversation_id), record_id TEXT NOT NULL '
                        'REFERENCES records(record_id), position INTEGER NOT NULL, '
                        'PRIMARY KEY(conversation_id, record_id), UNIQUE(conversation_id, position))')
            con.execute(f'PRAGMA user_version = {SCHEMA_VERSION}')
            con.commit()
        yield con
    finally:
        # 事务异常时close回滚，不删除或重建原数据库。
        con.close()


def initialize(path):
    with connection(path) as con:
        con.execute('SELECT record_id, revision, body FROM records LIMIT 0')
        con.execute('SELECT conversation_id, kb_id, version_id FROM conversations LIMIT 0')
        con.execute('SELECT position FROM conversation_records LIMIT 0')


def load_records(path):
    with connection(path) as con:
        rows = con.execute('SELECT record_id, revision, body FROM records ORDER BY seq').fetchall()
    records, revisions = [], {}
    for record_id, revision, body in rows:
        try:
            record = clean_record(json.loads(body))
        except (ValueError, TypeError, KeyError, HistoryError) as exc:
            raise HistoryError('历史记录损坏；请保留数据库，不要覆盖。') from exc
        if record['record_id'] != record_id:
            raise HistoryError('历史记录标识不一致；请保留数据库。')
        records.append(record)
        revisions[record_id] = revision
    return records, revisions


def save_record(path, record, expected_revision=None, conversation_id=None):
    record = clean_record(record)
    body, record_id = encode(record), record['record_id']
    snapshot = record['kb_snapshot']
    kb_id, version_id = snapshot.get('knowledge_base_id'), snapshot.get('knowledge_base_version')
    with connection(path) as con:
        con.execute('BEGIN IMMEDIATE')
        if conversation_id:
            if not kb_id or not version_id:
                raise HistoryError('记录缺少知识库/版本，不能绑定历史对话。')
            scope = con.execute('SELECT kb_id, version_id FROM conversations WHERE conversation_id=?',
                                (conversation_id,)).fetchone()
            if scope and scope != (kb_id, version_id):
                raise HistoryError('已阻止将不同知识库或资料版本写入同一历史对话。')
        old = con.execute('SELECT revision, body FROM records WHERE record_id=?', (record_id,)).fetchone()
        if old is None:
            con.execute('INSERT INTO records(record_id, revision, body) VALUES (?, 1, ?)', (record_id, body))
            revision = 1
        elif old[1] == body:
            revision = old[0]  # 同一记录重试/界面重跑不增加行或修订号。
        else:
            if expected_revision != old[0]:
                raise HistoryConflict('该记录已被另一页面更新；未覆盖磁盘评分，请先下载当前记录并重新加载核对。')
            prior = json.loads(old[1])
            immutable = lambda r: {k: v for k, v in r.items() if k not in ('case_id', 'manual_review')}
            if immutable(prior) != immutable(record):
                raise HistoryError('已阻止改写历史问题、答案、原文或批次；只允许补录题号和评分。')
            revision = old[0] + 1
            con.execute('UPDATE records SET revision=?, body=? WHERE record_id=?', (revision, body, record_id))
        if conversation_id:
            timestamp = now()
            con.execute('INSERT OR IGNORE INTO conversations VALUES (?, ?, ?, ?, ?, ?)',
                        (conversation_id, kb_id, version_id, record['question'][:80], timestamp, timestamp))
            linked = con.execute('SELECT position FROM conversation_records WHERE conversation_id=? AND record_id=?',
                                 (conversation_id, record_id)).fetchone()
            if not linked:
                position = con.execute('SELECT COALESCE(MAX(position), 0)+1 FROM conversation_records '
                                       'WHERE conversation_id=?', (conversation_id,)).fetchone()[0]
                con.execute('INSERT INTO conversation_records VALUES (?, ?, ?)', (conversation_id, record_id, position))
                con.execute('UPDATE conversations SET updated_at_utc=? WHERE conversation_id=?',
                            (timestamp, conversation_id))
        con.commit()
        return revision


def list_conversations(path, kb_id, version_id):
    with connection(path) as con:
        rows = con.execute('SELECT c.conversation_id, c.title, c.created_at_utc, c.updated_at_utc, '
            'COUNT(r.record_id) FROM conversations c JOIN conversation_records r '
            'ON c.conversation_id=r.conversation_id WHERE c.kb_id=? AND c.version_id=? '
            'GROUP BY c.conversation_id ORDER BY c.updated_at_utc DESC, c.conversation_id',
            (kb_id, version_id)).fetchall()
    return [dict(zip(('conversation_id', 'title', 'created_at_utc', 'updated_at_utc', 'record_count'), row))
            for row in rows]


def load_conversation(path, conversation_id, kb_id, version_id):
    with connection(path) as con:
        scope = con.execute('SELECT kb_id, version_id FROM conversations WHERE conversation_id=?',
                            (conversation_id,)).fetchone()
        if scope != (kb_id, version_id):
            raise HistoryError('所选历史不属于当前知识库及资料版本，已阻止恢复。')
        rows = con.execute('SELECT r.body FROM conversation_records cr JOIN records r '
            'ON cr.record_id=r.record_id WHERE cr.conversation_id=? ORDER BY cr.position',
            (conversation_id,)).fetchall()
    try:
        records = [clean_record(json.loads(row[0])) for row in rows]
    except (ValueError, TypeError, KeyError, HistoryError) as exc:
        raise HistoryError('历史对话记录损坏，已阻止恢复；请保留数据库。') from exc
    if any((r['kb_snapshot'].get('knowledge_base_id'), r['kb_snapshot'].get('knowledge_base_version'))
           != (kb_id, version_id) for r in records):
        raise HistoryError('历史对话记录范围不一致，已阻止恢复。')
    return records
