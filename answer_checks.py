"""有限数值证据疑点检查；不判定事实真伪，不改写答案或引用，不调用API。

只识别常用单位的数值、完整范围及部分约束词。不处理任意语义、单位换算、
产品/接口归属、表格布局或所有否定关系。无疑点不代表引用语义通过。
"""
import re
import json
from decimal import Decimal

VERSION = 'numeric-evidence-check-v1.1'
NUMBER = r'[+-]?\d+(?:\.\d+)?'
UNIT = r'(?:kWh|mWh|Wh|mAh|Ah|kW|mW|W|mV|V|mA|A|ms|us|μs|s|kg|g|mm|cm|m|℃|°C|°F|kHz|Hz)'
QUANTITY = re.compile(
    rf'(?<![A-Za-z0-9.])(?P<low>{NUMBER})\s*'
    rf'(?:(?P<first_unit>{UNIT})?\s*[-~～–—至到]\s*(?P<high>{NUMBER})\s*)?'
    rf'(?P<unit>{UNIT})(?![A-Za-z0-9])', re.I)


def compact(text):
    return re.sub(r'\s+', '', text).replace('−', '-').replace('﹣', '-')


def unit_key(unit):
    return unit.lower().replace('°c', '℃')


def number_key(number):
    return str(Decimal(number).normalize())


def quantities(text):
    results = []
    for m in QUANTITY.finditer(text):
        unit = unit_key(m['unit'])
        if m['first_unit'] and unit_key(m['first_unit']) != unit:
            continue
        before = compact(text[max(0, m.start()-14):m.start()])
        after = compact(text[m.end():m.end()+6])
        required = set()
        if not m['high']:
            if after.startswith(('以内', '内')) or re.search(r'(?:不超过|至多|最多|小于等于|≤)$', before):
                required.add('upper_bound')
            if re.search(r'(?:至少|不低于|大于等于|≥)$', before):
                required.add('lower_bound')
            if re.search(r'(?:大约|约|大概|≈)$', before):
                required.add('approximate')
        results.append({'key': (number_key(m['low']), number_key(m['high']) if m['high'] else None, unit),
                        'text': m.group(), 'required': required, 'start': m.start(), 'end': m.end()})
    return results


def segments(answer):
    # “12W。\n[1]”仍按前一句的引用检查；原答案不修改。
    working = re.sub(r'([。！？])\s*((?:\[\d+\]\s*)+)', r'\2\1', answer)
    return [s.strip() for s in re.split(r'[。！？；;\n]+', working) if s.strip()]


def inspect_answer(answer, sources, question=''):
    """返回疑点，不能作为自动通过评分。sources保持原始编号顺序。"""
    issues = []
    checked = 0
    for segment in segments(answer):
        ids = sorted({int(v) for v in re.findall(r'\[(\d+)\]', segment)})
        values = quantities(segment)
        if not ids or not values:
            continue
        valid = [i for i in ids if 1 <= i <= len(sources)]
        if not valid:
            continue  # 编号越界交由原格式检查；不把无效编号误报为数值证据缺失。
        found = {i: quantities(sources[i-1]['text']) for i in valid}
        keys = {v['key'] for v in values}
        for i, matches in found.items():
            if not any(v['key'] in keys for v in matches):
                issues.append({'code': 'citation_without_claim_quantity', 'source_id': i,
                               'message': f'引用[{i}]未包含本句所述数值或完整范围，请核对引用是否直接支持结论。'})
        for value in values:
            checked += 1
            # 用户给定的温度条件可用于与手册范围比较，不要求该测试温度也逐字写在手册里。
            # 只豁免明确的温度比较，不豁免问题中的产品参数断言或错误前提。
            tail = compact(segment[value['end']:value['end']+8])
            if (value['key'][2] in ('℃', '°f') and value['key'][1] is None
                    and any(v['key'] == value['key'] for v in quantities(question))
                    and tail.startswith(('低于', '高于', '不在', '不属于', '不符合', '不满足'))):
                continue
            matches = [v for vs in found.values() for v in vs if v['key'] == value['key']]
            if not matches:
                issues.append({'code': 'quantity_not_in_cited_fragments', 'quantity': value['text'],
                               'message': f'已引用片段未包含本句的完整数值/范围“{value["text"]}”，请从全部候选寻找直接依据，不要补猜缺失边界。'})
                continue
            required = set.intersection(*(v['required'] for v in matches))
            expressed = set(value['required'])
            missing = sorted(required - expressed)
            if missing:
                issues.append({'code': 'qualifier_missing', 'quantity': value['text'], 'qualifiers': missing,
                               'message': f'“{value["text"]}”的原文含限定{missing}，回答没有保留；按原文保留上限/下限/近似含义，不改成固定精确值。'})
    # 去掉同一句重复疑点，保留稳定顺序，限制重试提示长度。
    unique = []
    for issue in issues:
        if issue not in unique:
            unique.append(issue)
    return {'version': VERSION, 'checked_quantities': checked,
            'issues': unique[:12], 'status': 'suspected_issue' if unique else 'no_issue_detected',
            'limitation': '有限数值检查；无疑点不代表事实、产品/接口归属或引用语义通过。'}


def correction_hints(answer, sources, question=''):
    """只定位同文件的数值原文行，不判断接口/产品语义，不替模型改引用。"""
    hints = []
    for segment in segments(answer):
        if not inspect_answer(segment, sources, question)['issues']:
            continue
        ids = {int(v) for v in re.findall(r'\[(\d+)\]', segment) if 1 <= int(v) <= len(sources)}
        filenames = {sources[i-1].get('label', '').split(' · PDF第')[0] for i in ids}
        if not filenames:
            continue
        for value in quantities(segment):
            count = 0
            for i, source in enumerate(sources, 1):
                label = source.get('label', '')
                if label.split(' · PDF第')[0] not in filenames:
                    continue
                matches = [v for v in quantities(source['text']) if v['key'] == value['key']]
                if not matches:
                    continue
                m = matches[0]
                text = source['text']
                start = text.rfind('\n', 0, m['start']) + 1
                end = text.find('\n', m['end'])
                if end < 0:
                    end = len(text)
                # 极长行仅给包含数值的窗口；仍是该片段中的原始连续子串。
                start = max(start, m['start']-80)
                end = min(end, start+180)
                hint = {'quantity': value['text'], 'source_id': i, 'label': label,
                        'source_line': text[start:end]}
                if hint not in hints:
                    hints.append(hint)
                    count += 1
                if len(hints) >= 8:
                    return hints
                if count >= 3:
                    break
    return hints


def supported_quantity_keys(answer, sources):
    """保留首次输出中已有引用包含的数值项目；这不是语义支持证明。"""
    values = {}
    for segment in segments(answer):
        ids = {int(v) for v in re.findall(r'\[(\d+)\]', segment) if 1 <= int(v) <= len(sources)}
        cited_keys = {v['key'] for i in ids for v in quantities(sources[i-1]['text'])}
        for v in quantities(segment):
            if v['key'] in cited_keys:
                values[v['key']] = v['text']
    return values


def correction_decision(initial, candidate, sources, question='', *, candidate_is_refusal=False):
    """仅按可检测疑点采纳，不能证明新答案语义更正确。不会修改任何文本。"""
    before = inspect_answer(initial, sources, question)
    after = inspect_answer(candidate, sources, question)
    def flags(answer, check, refusal=False):
        result = {(i['code'], i.get('source_id'), i.get('quantity'), tuple(i.get('qualifiers', [])))
                  for i in check['issues']}
        refs = {int(v) for v in re.findall(r'\[(\d+)\]', answer)}
        if not refs and not refusal:
            result.add(('missing_citation', None, None, ()))
        result.update(('invalid_citation', i, None, ()) for i in refs if not 1 <= i <= len(sources))
        return result
    old_flags = flags(initial, before)
    new_flags = flags(candidate, after, candidate_is_refusal)
    candidate_keys = {v['key'] for v in quantities(candidate)}
    protected = supported_quantity_keys(initial, sources)
    removed = [text for key, text in protected.items() if key not in candidate_keys]
    accepted, reason = False, 'no_strict_improvement'
    if not candidate.strip():
        reason = 'empty_response'
    elif candidate_is_refusal and not new_flags:
        accepted, reason = True, 'conservative_refusal'
    elif removed:
        reason = 'supported_numeric_items_removed'
    elif new_flags - old_flags:
        reason = 'new_detectable_issue'
    elif len(new_flags) < len(old_flags):
        accepted, reason = True, 'detectable_issues_reduced'
    return {'accepted': accepted, 'reason': reason, 'before_issue_count': len(old_flags),
            'after_issue_count': len(new_flags), 'removed_supported_quantities': removed}, after


def format_correction_hints(hints):
    return '\n'.join(json.dumps(hint, ensure_ascii=False) for hint in hints)
