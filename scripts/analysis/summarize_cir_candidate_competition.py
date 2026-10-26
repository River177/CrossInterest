"""Report accepted candidate competition diagnostics without downloading predictions."""
import csv
import math
from pathlib import Path

from read_cir_candidate_competition import read, write, sha, require


def main():
    manifest=Path('manifests/mind_cir_candidate_competition_20260924_v1.json')
    m=read(manifest);base=Path('artifacts/mind-cir-candidate-competition-20260924')
    values={};receipts={};source_files=[]
    for arm,j in m['jobs'].items():
        folder=base/'final-readback'/m['experiment']/j['job']
        receipt=read(folder/'independent_acceptance.json')
        require(receipt['status']=='independently_accepted','Arm is not accepted')
        native=read(j['native_receipt'])
        require(native['native_status']=='Completed','Native job not complete')
        values[arm]=read(folder/'complete.json');receipts[arm]=receipt
        source_files += [folder/'complete.json',folder/'independent_acceptance.json',folder/'download_receipt.json',Path(j['native_receipt'])]
    comparisons=[]
    for k in (1,5,10,20):
        for metric in ('recall','ndcg'):
            key=f'interest_macro_{metric}_at_{k}'
            before={a:receipts[a]['metrics']['original'][key] for a in values}
            after={a:receipts[a]['metrics']['unseen'][key] for a in values}
            gap0=before['cir5_ni3']-before['ni_only'];gap1=after['cir5_ni3']-after['ni_only']
            comparisons.append({'metric':key, 'ni_only_original':before['ni_only'],'cir_original':before['cir5_ni3'],
                                'ni_only_unseen':after['ni_only'],'cir_unseen':after['cir5_ni3'],
                                'original_gap_pp':100*gap0,'unseen_gap_pp':100*gap1,'gap_change_pp':100*(gap1-gap0)})
    out=base/'final-report';out.mkdir(parents=True,exist_ok=True)
    with (out/'comparison.csv').open('w') as f:
        writer=csv.DictWriter(f,fieldnames=list(comparisons[0]));writer.writeheader();writer.writerows(comparisons)
    report={'status':'both_independently_accepted','experiment':m['experiment'],'comparison':comparisons,
            'historical_prediction_slot_fraction':{a:v['historical_prediction_slot_fraction'] for a,v in values.items()},
            'target_ranks':{a:{t:{k:d[k] for k in ('mean_target_rank','class_macro_mean_target_rank')} for t,d in v['treatments'].items()} for a,v in values.items()},
            'tie_boundary_counts':{a:v['positive_occurrences_tied_with_cutoff_score'] for a,v in values.items()},
            'scope':m['contract']['comparison_limit']}
    write(out/'comparison.json',report)
    lines=['# CIR 候选竞争诊断结果','',
           '两个 evaluation-only 作业已独立验收，原生状态均为 Completed。两组各 31,392 条未过滤 Top20 的标签与分数完全复现各自已接受预测；随后从同一份完整 108 类分数构造历史过滤排名。','',
           '结果支持：CIR 确实减少了历史类别对 TopK 名额的占用，但其相对优势在完全排除这些候选后仍然存在，尤其 Recall/NDCG@10 的差距变化很小。历史候选竞争不足以解释这组实验的全部收益；未见类别之间的排序差异仍是下一步需要解释的现象。','',
           '表内指标为百分数，差值为 CIR5→NI3 减 NI-only 的百分点。','',
           '| 指标 | NI-only 原始 | CIR 原始 | 原始差值 | NI-only 过滤 | CIR 过滤 | 过滤后差值 |',
           '|---|---:|---:|---:|---:|---:|---:|']
    for d in comparisons:
        name=d['metric'].replace('interest_macro_','').replace('_at_','@')
        lines.append(f"| {name} | {100*d['ni_only_original']:.4f} | {100*d['cir_original']:.4f} | {d['original_gap_pp']:+.4f} | {100*d['ni_only_unseen']:.4f} | {100*d['cir_unseen']:.4f} | {d['unseen_gap_pp']:+.4f} |")
    lines+=['','## 历史类别竞争与正例排名','',
            '| 模型 | 原 Top10 历史类别占比 | 原 Top20 历史类别占比 | 原始平均正例排名 | 过滤后平均正例排名 | 过滤后类宏平均排名 |',
            '|---|---:|---:|---:|---:|---:|']
    for arm,v in values.items():
        h=v['historical_prediction_slot_fraction'];t=v['treatments']
        lines.append(f"| {arm} | {h['10']*100:.4f}% | {h['20']*100:.4f}% | {t['original']['mean_target_rank']:.4f} | {t['unseen']['mean_target_rank']:.4f} | {t['unseen']['class_macro_mean_target_rank']:.4f} |")
    per_class_rows=[]
    for c in values['ni_only']['per_class_support']:
        for k in (1,5,10,20):
            for metric,field in [('recall','hits'),('ndcg','discounted_gain')]:
                row={'class':c,'support':values['ni_only']['per_class_support'][c],'metric':f'{metric}@{k}'}
                for arm,v in values.items():
                    for t,d in v['treatments'].items():
                        cell=d['per_class'][c];row[f'{arm}_{t}']=cell[f'{field}_at_{k}']/cell['support']
                per_class_rows.append(row)
    with (out/'per_class_comparison.csv').open('w') as f:
        writer=csv.DictWriter(f,fieldnames=list(per_class_rows[0]));writer.writeheader();writer.writerows(per_class_rows)
    lines+=['','平均正例排名按 37,728 次目标出现计；类宏平均先在每个目标类内求均值，再平均 104 个支持类。历史类别占比的分母为记录数×K。过滤后的历史类别占比为零。','',
            '## 解释范围','',
            '比较的是同一批用户、同一输入、同一正例与模型下的候选集合干预。过滤前后 CIR−NI-only 差距的变化可检验历史类别竞争是否与优势有关；仍然保留的差距体现未见候选之间的排序差异。不能把差距缩小比例解释为可加的因果贡献，也不能用它定位知识唯一存储于 LoRA 或 SID。','',
            'NI-only 保留历史单卡转四卡训练和 FLA 执行差异。补评分分别沿用 NI-only 的 FLA/batch8 和 CIR 的 Torch fallback/batch2，以精确复现原结果；这仍是机制筛查，不能作为严格匹配、跨 seed 的因果结论。','',
            '同分处理固定为原生 Top20 前缀，再按分数与 mapping index 排序剩余 88 类。边界同分计数保存在 comparison.json；同分次序是该诊断的评估约定。','',
            '下一步建议（未执行）：保持可见历史类别序列不变，只将标题替换为其他同类别标题，对两个已接受模型再次做成对评价。它可以进一步区分类别组合信息与标题内容对当前优势的作用；本次没有新增该条件。','',
            '## 验收与来源','',
            'Full Dev 31,392 条、37,728 个正例、104 支持类，四分片各 7,848 条；全部 108 类分数有限，原预测逐条 hash/输入绑定，正例删除 0，过滤后最少 73 候选。既有 metric library、远端独立计数及本地从每类完整 rank 直方图重算的指标一致。训练更新数为零。','',
            f"- Manifest：`{manifest}`",
            f"- 冻结 bundle SHA：`{m['bundle_sha256']}`",
            f"- CSV：`{out/'comparison.csv'}`",
            f"- 紧凑读回：`{base/'final-readback'/m['experiment']}`",
            '- 完整 108 类分数、逐正例排名明细及权重留远端。没有创建自动监控。']
    for arm,j in m['jobs'].items():
        lines.append(f"- {arm}: backend `{j['backend_id']}`, prefix `{j['results_prefix']}`, adapter SHA `{m['contract']['models'][arm]['adapter_file_sha256']}`")
    path=Path('docs/experiments/2026-09-24/cir_candidate_competition_results.md');path.write_text('\n'.join(lines)+'\n')
    m.update(status='both_independently_accepted',report=str(path),comparison_json=str(out/'comparison.json'))
    for arm in values:
        m['jobs'][arm]['status']='independently_accepted'
    write(manifest,m)
    source_files += [manifest,path,out/'comparison.json',out/'comparison.csv',out/'per_class_comparison.csv']
    write(out/'delivery_sha256.json',{str(p):sha(p) for p in source_files})
    print('\n'.join(lines[:18]))


if __name__=='__main__':main()
