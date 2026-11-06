"""v6.6.1无API自检，不读.env或真实知识库/历史，只用临时数据库。"""
import ast
import copy
from pathlib import Path
from tempfile import TemporaryDirectory
import answer_checks as guard
import evaluation_tools as evaluation
import history_storage as history


def checks():
    sources=[{'label':'alpha.pdf · PDF第2页','text':'车充输入最大功率85W。'},
             {'label':'alpha.pdf · PDF第2页','text':'车充输出最大功率119W。'},
             {'label':'other.pdf · PDF第2页','text':'车充输出最大功率119W。'}]
    first='输入85W[1]。输出119W[1][2]。'
    bad='输入85W[1]。输出119W[1]。'
    good='输入85W[1]。输出119W[2]。'
    saved=copy.deepcopy(sources)
    hints=guard.correction_hints(first,sources)
    assert len(hints)==1 and hints[0]['source_id']==2 and hints[0]['source_line']==sources[1]['text']
    assert not guard.correction_decision(first,bad,sources)[0]['accepted']
    decision,check=guard.correction_decision(first,good,sources)
    assert decision['accepted'] and not check['issues']
    assert not guard.correction_decision(first,first,sources)[0]['accepted']
    assert not guard.correction_decision(first,'输入85W[1]。',sources)[0]['accepted']
    assert guard.correction_decision(first,'当前资料不足，无法确认。',sources,candidate_is_refusal=True)[0]['accepted']
    assert guard.correction_decision('输入85W','输入85W[1]',sources)[0]['accepted']
    assert not guard.correction_decision(first,good+'参数来自[9]。',sources)[0]['accepted']
    assert guard.inspect_answer('18ms[1]',[{'text':'18ms内切换。'}])['issues']
    assert sources==saved
    assert evaluation.APP_VERSION=='6.6.1-correction-selection'
    assert guard.VERSION=='numeric-evidence-check-v1.1'
    tree=ast.parse(Path(__file__).with_name('app.py').read_text(encoding='utf-8-sig'))
    const={n.targets[0].id:ast.literal_eval(n.value) for n in tree.body if isinstance(n,ast.Assign)
           and isinstance(n.targets[0],ast.Name) and n.targets[0].id in ('PROMPT_VERSION','RETRIEVAL_VERSION')}
    assert const=={'PROMPT_VERSION':'6','RETRIEVAL_VERSION':'v6.6-unit-field-hybrid-neighbors'}
    assert any(isinstance(n,ast.Call) and isinstance(n.func,ast.Attribute)
               and isinstance(n.func.value,ast.Name) and n.func.value.id=='answer_checks'
               and n.func.attr=='correction_decision' for n in ast.walk(tree)), 'app.py尚未替换为v6.6.1。'
    stage={'stage':'correction','answer':good,'check':check,'decision':decision,'retry_question':'定位原文行'}
    r=evaluation.new_record('输入输出？',good,sources,[],{'answer_checks':[stage]},
        {'knowledge_base_id':'temp','knowledge_base_version':'v','run_id':'temp-run'})
    with TemporaryDirectory(prefix='rag_v661_check_') as t:
        p=Path(t)/'temp.sqlite3'
        history.save_record(p,r,conversation_id='temp-thread')
        loaded,_=history.load_records(p)
        assert loaded[0]['telemetry']['answer_checks']==[stage]
    return 12


if __name__=='__main__':
    print(f'v6.6.1离线自检通过：{checks()}项。')
    print('未调用API；仅使用并删除临时测试数据库，未访问原知识库与历史。')
