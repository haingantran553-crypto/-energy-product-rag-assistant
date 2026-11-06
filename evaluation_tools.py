"""问答证据和人工评测记录；仅使用Python标准库，不调用模型。

命令行汇总：python evaluation_tools.py qa_records.json
"""

import argparse
import copy
import json
from datetime import datetime, timezone
from pathlib import Path
from statistics import median
from uuid import uuid4
import re


SCHEMA_VERSION = "1.0"
APP_VERSION = "6.6.1-correction-selection"
REVIEW_FIELDS = ("fact", "citation_support", "retrieval_coverage")
REVIEW_OPTIONS = ("待核验", "通过", "未通过")


def citation_audit(answer, sources):
    """只检查[数字]是否存在和越界，不判断事实或引用语义。"""
    numbers = sorted({int(value) for value in re.findall(r"\[(\d+)\]", answer)})
    return {
        "valid_ids": [value for value in numbers if 1 <= value <= len(sources)],
        "invalid_ids": [value for value in numbers if value < 1 or value > len(sources)],
        "has_numeric_citation": bool(numbers),
    }


def new_record(question, answer, sources, notes, telemetry, kb_snapshot,
               status="success", error_type=None):
    return {
        "record_id": uuid4().hex,
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "case_id": "",
        "status": status,
        "question": question,
        "answer": answer,
        "sources": [dict(copy.deepcopy(source), id=index)
                    for index, source in enumerate(sources, start=1)],
        "notes": list(notes),
        "telemetry": copy.deepcopy(telemetry),
        "citation_audit": citation_audit(answer, sources),
        "kb_snapshot": copy.deepcopy(kb_snapshot),
        "manual_review": {"fact": "待核验", "citation_support": "待核验",
                          "retrieval_coverage": "待核验", "comment": ""},
        "error_type": error_type,
    }


def export_json(records):
    return json.dumps({
        "schema_version": SCHEMA_VERSION,
        "exported_at_utc": datetime.now(timezone.utc).isoformat(),
        "measurement_notes": {
            "total_seconds": "应用内检索与生成耗时；不含用户阅读和核验时间",
            "generation_seconds": "包含最多一次格式/数值疑点校正；不是模型服务端推理时间",
            "answer_checks": "有限数值疑点检查与校正前后原始输出；不自动判定事实或引用语义正确",
            "citation_audit": "只检查数字编号，不证明结论正确或引用支持",
            "manual_review": "默认待核验；不得据此报告全量通过率",
            "storage": "v6.4新增记录与评分写入本地历史；若保存失败仍保留当前会话并提示下载。旧JSON不自动导入。",
        },
        "records": records,
    }, ensure_ascii=False, indent=2, allow_nan=False)


def review_counts(records, field, allow_na=False):
    counts = {"通过": 0, "未通过": 0, "待核验": 0, "不适用": 0}
    for record in records:
        value = record.get("manual_review", {}).get(field, "待核验")
        if value not in counts or (value == "不适用" and not allow_na):
            value = "待核验"
        counts[value] += 1
    applicable = counts["通过"] + counts["未通过"]
    # 未核验时不给出完整通过率，避免只算已通过子集。
    counts["full_rate"] = (counts["通过"] / applicable
                           if not counts["待核验"] and applicable else None)
    return counts


def overall_status(record):
    if record.get("status") != "success":
        return "请求失败"
    review = record.get("manual_review", {})
    fact, citation = review.get("fact"), review.get("citation_support")
    if fact == "未通过" or citation == "未通过":
        return "未通过"
    if fact == "通过" and citation in ("通过", "不适用"):
        return "通过"
    return "待核验"


def _percentile(values, percent):
    """线性插值百分位；小样本报告保留样本量。"""
    values = sorted(values)
    position = (len(values) - 1) * percent
    low = int(position)
    high = min(low + 1, len(values) - 1)
    return values[low] + (values[high] - values[low]) * (position - low)


def summarize_records(records):
    """按每次建库分组，避免把不同资料/配置结果混成一个准确率。"""
    groups = {}
    for record in records:
        snapshot = record.get("kb_snapshot", {})
        key = snapshot.get("run_id", "unknown")
        groups.setdefault(key, []).append(record)
    summaries = []
    for run_id, group in groups.items():
        success = [item for item in group if item.get("status") == "success"]
        clarifications = [item for item in success
                          if (item.get("question_resolution") or {}).get("status") == "clarification"]
        qa_success = [item for item in success if item not in clarifications]
        latencies = [item.get("telemetry", {}).get("total_seconds") for item in qa_success]
        latencies = [value for value in latencies
                     if isinstance(value, (int, float)) and value >= 0]
        overall = {name: 0 for name in ("通过", "未通过", "待核验", "请求失败")}
        for item in group:
            if item in clarifications:
                continue
            overall[overall_status(item)] += 1
        summaries.append({
            "run_id": run_id,
            "snapshot": copy.deepcopy(group[0].get("kb_snapshot", {})),
            "requests": len(group), "success": len(success),
            "clarification_requests": len(clarifications), "qa_success": len(qa_success),
            "failed": len(group) - len(success),
            "fact": review_counts(qa_success, "fact"),
            "citation_support": review_counts(qa_success, "citation_support", allow_na=True),
            "retrieval_coverage": review_counts(qa_success, "retrieval_coverage"),
            "overall": overall,
            "latency_samples": len(latencies),
            "median_seconds": median(latencies) if latencies else None,
            "p95_seconds": _percentile(latencies, 0.95) if latencies else None,
            "retry_requests": sum(item.get("telemetry", {}).get("citation_retry_count", 0) > 0
                                  for item in group),
        })
    return summaries


def render_report(records):
    lines = ["# 问答评测汇总", "",
             "本报告依据导出记录和人工评分生成；未核验项不计为通过。",
             "按每次建库分组；重问题目均保留，不能把重复尝试当作不同测试题。", ""]
    for group in summarize_records(records):
        snapshot = group["snapshot"]
        lines.extend([
            "## 建库批次 " + group["run_id"], "",
            "- 应用版本：" + str(snapshot.get("app_version", "未知")),
            "- 代码SHA256：" + str(snapshot.get("app_sha256", "未知")),
            "- 聊天模型：" + str(snapshot.get("chat_model", "未知")),
            "- 向量模型：" + str(snapshot.get("embedding_model", "未知")),
            "- 有效资料：" + "、".join(snapshot.get("active_sources", [])),
            f"- 请求：{group['requests']}；成功返回：{group['success']}；请求失败：{group['failed']}",
            f"- 其中资料问答成功：{group['qa_success']}；规则澄清：{group['clarification_requests']}（不计入下方问答评分和耗时）",
            f"- 启动引用补全的请求数：{group['retry_requests']}", "",
            "| 人工核验项 | 通过 | 未通过 | 待核验 | 不适用 | 完整通过率 |",
            "|---|---:|---:|---:|---:|---|",
        ])
        for field, label in (("fact", "事实/任务"), ("citation_support", "引用支持"),
                             ("retrieval_coverage", "检索证据覆盖")):
            counts = group[field]
            rate = (f"{counts['full_rate']:.1%}" if counts["full_rate"] is not None
                    else "未报告（有待核验项或无适用样本）")
            lines.append(f"| {label} | {counts['通过']} | {counts['未通过']} | "
                         f"{counts['待核验']} | {counts['不适用']} | {rate} |")
        overall = group["overall"]
        lines.extend(["", f"总体：通过{overall['通过']}；未通过{overall['未通过']}；"
                      f"待核验{overall['待核验']}；请求失败{overall['请求失败']}。", ""])
        if group["latency_samples"]:
            lines.append(f"成功请求耗时：n={group['latency_samples']}，"
                         f"中位数{group['median_seconds']:.2f}秒，"
                         f"P95 {group['p95_seconds']:.2f}秒（小样本仅供描述）。")
        else:
            lines.append("暂无有效成功请求耗时样本。")
        case_ids = [item.get("case_id", "").strip() for item in records
                    if item.get("kb_snapshot", {}).get("run_id", "unknown") == group["run_id"]]
        if any(not case_id for case_id in case_ids):
            lines.append("部分记录未填题号，不能据请求数宣称固定评测集已完成。")
        if len([value for value in case_ids if value]) != len(set(value for value in case_ids if value)):
            lines.append("存在重复题号，请在比较时区分首次结果和复测，不覆盖失败记录。")
        lines.append("")
    lines.extend(["说明：总体通过要求事实/任务通过，且引用支持通过或经人工判定不适用。",
                  "纯资料不足回答的引用可判不适用；若补充具体事实，仍需检查其引用。",
                  "耗时不含人工查证时间，不能直接换算为业务效率提升。", ""])
    return "\n".join(lines)


def main():
    parser = argparse.ArgumentParser(description="汇总应用导出的人工评测JSON")
    parser.add_argument("file", type=Path)
    args = parser.parse_args()
    payload = json.loads(args.file.read_text(encoding="utf-8-sig"))
    print(render_report(payload["records"]))


if __name__ == "__main__":
    main()
