"""有限规则补全省略问题。不调用模型，不读取历史答案，不提供参数值。"""
import re

VERSION = 'bounded-question-context-v1.1'
FIELDS = ('容量', '功率', '电压', '电流', '净重', '重量', '尺寸', '型号',
          '电芯材料', '电池材料', '温度', '寿命', '售价')
INTERFACES = ('USB-C', 'USB-A', '车充', '太阳能', '交流')
DIRECTIONS = ('输入', '输出')
QUANTITIES = ('功率', '电压', '电流')
PRONOUNS = ('它', '该产品', '这款', '该型号', '这个产品')


def products_from_chunks(kb):
    """只识别短ASCII含数字封面名，或TXT明确的产品名称行；不能识别则不猜。"""
    products = {}
    for doc in kb.get('chunk_lookup', {}).values():
        source = doc.metadata.get('source')
        if not source:
            continue
        lines = [line.strip() for line in doc.page_content.splitlines() if line.strip()]
        if not lines:
            continue
        line = None
        if doc.metadata.get('page') == 0:
            line = lines[0]
        elif doc.metadata.get('kind') == 'txt' and doc.metadata.get('start_index', 0) == 0:
            match = re.match(r'^产品名称\s*[:：]\s*(.+)$', lines[0])
            if match:
                line = match.group(1)
        if not line or len(line) > 80:
            continue
        line = re.sub(r'^ECOFLOW\s+', '', line, flags=re.I)
        if not re.fullmatch(r'[A-Za-z][A-Za-z0-9 ._-]*\d[A-Za-z0-9 ._-]*', line):
            continue
        name = re.sub(r'\s+', ' ', line).strip()
        key = re.sub(r'[\s_-]', '', name).casefold()
        entry = products.setdefault(key, {'name': name, 'source_files': []})
        if source not in entry['source_files']:
            entry['source_files'].append(source)
    return list(products.values())


def name_pattern(name):
    tokens = re.findall(r'[A-Za-z]+|\d+|\.', name)
    return r'(?<![A-Za-z0-9])' + r'[\s_-]*'.join(re.escape(t) for t in tokens) + r'(?![A-Za-z0-9])'


def mentions(text, products):
    hits, occupied, unknown_suffix = [], [], False
    for product in sorted(products, key=lambda p: -len(p['name'])):
        for match in re.finditer(name_pattern(product['name']), text, flags=re.I):
            span = match.span()
            if any(span[0] < end and start < span[1] for start, end in occupied):
                continue
            suffix = re.match(r'[\s_-]+([A-Za-z][A-Za-z0-9-]*)', text[span[1]:])
            if suffix and suffix.group(1).upper() not in ('USB-C', 'USB-A', 'EPS', 'UPS'):
                unknown_suffix = True
                continue
            hits.append({'name': product['name'], 'span': span, 'alias': False})
            occupied.append(span)
    # 仅将唯一/可能歧义的英文末尾词作为短称，数字不作为产品短称。
    aliases = {}
    for product in products:
        suffix = product['name'].split()[-1]
        if re.fullmatch(r'[A-Za-z]{2,}', suffix) and suffix.casefold() != product['name'].casefold():
            aliases.setdefault(suffix.casefold(), []).append(product['name'])
    for alias, names in aliases.items():
        for match in re.finditer(name_pattern(alias), text, flags=re.I):
            span = match.span()
            if any(span[0] < end and start < span[1] for start, end in occupied):
                continue
            for name in names:
                hits.append({'name': name, 'span': span, 'alias': True})
    return hits, unknown_suffix


def slots(text, options):
    return [value for value in options if value.casefold() in text.casefold()]


def replace_spans(text, hits, replacement):
    for start, end in sorted({h['span'] for h in hits}, reverse=True):
        text = text[:start] + replacement + text[end:]
    return text


def bare_switch(text):
    for phrase in ('请标注资料引用', '请标注引用', '请引用', '换成', '那么', '另一款',
                   '另一个型号', '另外', '请问', '怎么样', '如何', '还有', '那', '换'):
        text = text.replace(phrase, '')
    return not re.sub(r'[\s，。？！?!、:：的呢]', '', text)


def strip_leading(text):
    text = re.sub(r'\s+', ' ', text).strip()
    return re.sub(r'^(?:(?:那么|另外|还有|请问|那|它|该产品|这款|该型号|这个产品|的)\s*)+', '', text)


def resolve(question, messages, products, *, kb_id, version_id, conversation_id, blocked=False):
    template_adjusted = False
    trace = {'version': VERSION, 'status': 'standalone', 'original_question': question,
             'retrieval_question': question, 'used_record_ids': [], 'considered_record_id': None,
             'conversation_id': conversation_id, 'knowledge_base_id': kb_id,
             'knowledge_base_version': version_id, 'rule': 'standalone',
             'reason': '完整问题按原文检索。', 'clarification': None}

    def clarify(rule, reason):
        trace.update(status='clarification', retrieval_question=None, rule=rule, reason=reason,
                     used_record_ids=[], clarification=reason + '请写明产品全名和要查的参数；涉及接口时注明输入或输出。')
        return trace

    def resolved(text, rule, reason, prior=None):
        if template_adjusted:
            reason += '上一轮旧版检索问题的重复主语/空白仅在本轮工作副本中规范化，原记录未修改。'
        trace.update(status='resolved', retrieval_question=text, rule=rule, reason=reason,
                     used_record_ids=[prior['record_id']] if prior else [])
        return trace

    current, unknown = mentions(question, products)
    current_names = sorted({h['name'] for h in current})
    fields = slots(question.replace('另一个型号', ''), FIELDS)
    # 明确写全产品与字段时不沿用历史，保留既有单轮行为。
    if fields and any(not h['alias'] for h in current) and not any(p in question for p in PRONOUNS):
        return trace
    short_alias = any(h['alias'] for h in current)
    is_followup = (short_alias or any(p in question for p in PRONOUNS)
                   or any(p in question for p in ('另一款', '另一个型号'))
                   or bool(re.match(r'^\s*(?:那|那么|换成|另外|还有)', question))
                   or (bool(re.search(r'呢[？?。\s]*$', question)) and not current_names))
    if not is_followup:
        return trace
    allowed_words = {'USB-C', 'USB-A', 'EPS', 'UPS', 'WH', 'KWH', 'MWH', 'MAH', 'AH',
                     'W', 'V', 'A', 'MS', 'C', 'ECOFLOW'}
    for token in re.finditer(r'[A-Za-z][A-Za-z0-9._-]*', question):
        if (not any(h['span'][0] <= token.start() and token.end() <= h['span'][1] for h in current)
                and token.group().upper() not in allowed_words):
            return clarify('unrecognized_name', '追问中有未确认的名称或缩写，不能套用上一轮产品。')
    if unknown:
        return clarify('unknown_product_suffix', '当前写法含未确认的产品后缀，不能按已有相近型号补全。')
    if len(current_names) > 1:
        return clarify('ambiguous_current_products', '这条省略问题涉及多个可能产品，无法唯一补全。')
    if any(term in question for term in ('对比', '比较', '分别', '两款')):
        return clarify('complex_followup', '本版不自动补全省略了型号或字段的比较问题。')
    if '温度' in fields and not any(t in question for t in ('充电', '放电', '存储', '储存', '最佳')):
        return clarify('missing_temperature_kind', '需要明确充电、放电、存储或最佳使用温度。')

    # 独立短称问法：目录中唯一的Max + 完整参数，不需要上一轮。
    if len(current_names) == 1 and fields and not any(p in question for p in PRONOUNS):
        if (not slots(question, QUANTITIES)
                or (len(slots(question, INTERFACES)) == 1 and len(slots(question, DIRECTIONS)) == 1)):
            return resolved(replace_spans(question, current, current_names[0]),
                            'unique_alias', '由当前资料的明确产品名展开唯一短称。')

    if blocked or not messages:
        return clarify('no_usable_history', '这条追问缺少可用的上一轮问题。')
    prior = messages[-1]  # 不越过失败、澄清或话题边界寻找更早问题。
    trace['considered_record_id'] = prior.get('record_id')
    snapshot = prior.get('kb_snapshot', {})
    previous_trace = prior.get('question_resolution') or {}
    if (prior.get('status') != 'success' or previous_trace.get('status') == 'clarification'
            or snapshot.get('knowledge_base_id') != kb_id
            or snapshot.get('knowledge_base_version') != version_id
            or (previous_trace.get('conversation_id') and previous_trace['conversation_id'] != conversation_id)):
        return clarify('history_scope_or_status', '上一轮不属于当前可用对话范围，或上一轮仍需澄清。')
    previous = prior.get('retrieval_question') or prior['question']
    previous_hits, previous_unknown = mentions(previous, products)
    previous_names = sorted({h['name'] for h in previous_hits})
    if previous_unknown or len(previous_names) != 1:
        return clarify('ambiguous_previous_product', '上一轮不能唯一确定产品，本版不从多产品比较中自动选择指代。')
    # 只处理已知v1补全器留下的“明确产品的\t它的…”格式，不改历史数据，也不修猜测性的指代。
    if (previous_trace.get('status') == 'resolved'
            and previous_trace.get('version') == 'bounded-question-context-v1'):
        working = re.sub(r'\s+', ' ', previous).strip()
        pattern = name_pattern(previous_names[0]) + r'\s*的\s*(?:它|该产品|这款|该型号|这个产品)\s*的'
        working = re.sub(pattern, lambda _: previous_names[0] + '的', working, flags=re.I)
        template_adjusted = working != previous
        previous = working
        previous_hits, _ = mentions(previous, products)
    if not slots(previous, FIELDS):
        return clarify('missing_previous_topic', '上一轮没有可明确继承的参数字段。')
    target = current_names[0] if current_names else previous_names[0]
    if any(p in question for p in ('另一款', '另一个型号')):
        others = [p['name'] for p in products if p['name'] != previous_names[0]]
        if len(others) != 1:
            return clarify('ambiguous_other_product', '当前资料不能唯一确定“另一款”。')
        target = others[0]
    without_products = replace_spans(question, current, '')
    if not fields and bare_switch(without_products):
        if not current_names and not any(p in question for p in ('另一款', '另一个型号')):
            return clarify('missing_target', '这条追问没有明确的新产品或新参数。')
        return resolved(replace_spans(previous, previous_hits, target), 'switch_product',
                        '继承当前对话上一轮问题，只替换已确认产品，不读取旧答案。', prior)

    interfaces, directions = slots(question, INTERFACES), slots(question, DIRECTIONS)
    prior_interfaces, prior_directions = slots(previous, INTERFACES), slots(previous, DIRECTIONS)
    prior_quantities = slots(previous, QUANTITIES)
    if not fields and (interfaces or directions):
        # “那输入呢？”必须有唯一旧接口、方向和功率/电压/电流字段。
        if (len(prior_interfaces) != 1 or len(prior_directions) != 1 or len(prior_quantities) != 1
                or len(interfaces) > 1 or len(directions) > 1):
            return clarify('missing_metric_context', '无法唯一继承接口、输入输出方向和功率/电压/电流字段。')
        residue = strip_leading(without_products)
        for word in interfaces + directions:
            residue = re.sub(re.escape(word), '', residue, flags=re.I)
        if not bare_switch(residue):
            return clarify('unsupported_short_form', '这条追问包含本版尚不能明确补全的条件。')
        text = replace_spans(previous, previous_hits, target)
        if interfaces:
            text = re.sub(re.escape(prior_interfaces[0]), interfaces[0], text, flags=re.I)
        if directions:
            text = text.replace(prior_directions[0], directions[0])
        return resolved(text, 'switch_interface_or_direction',
                        '继承上一轮的唯一接口/字段，只按本轮明确内容替换。', prior)
    if not fields:
        return clarify('missing_metric', '这条追问没有明确要查询的参数。')
    body = strip_leading(without_products).strip(' ，。？！?!呢')
    quantities = slots(question, QUANTITIES)
    if quantities:
        if len(quantities) != 1 or len(interfaces) > 1 or len(directions) > 1:
            return clarify('ambiguous_metric', '功率/电压/电流、接口或方向仍不唯一。')
        same_quantity = quantities == prior_quantities
        if not interfaces:
            if not same_quantity or len(prior_interfaces) != 1:
                return clarify('missing_interface', '需要明确查询哪个接口的参数。')
            body = prior_interfaces[0] + body
        if not directions:
            if not same_quantity or len(prior_directions) != 1:
                return clarify('missing_direction', '需要明确输入或输出方向。')
            body = prior_directions[0] + body
    if '温度' in fields and not any(t in question for t in ('充电', '放电', '存储', '储存', '最佳')):
        return clarify('missing_temperature_kind', '需要明确充电、放电、存储或最佳使用温度。')
    text = f'{target}的{body}？请按当前资料回答并标注引用。'
    return resolved(text, 'field_with_subject', '从上一轮问题补足唯一产品；参数以本轮明确内容为准。', prior)
