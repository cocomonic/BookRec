"""Generate documentation strictly from completed local experiment artifacts."""
import json
from pathlib import Path

ROOT=Path(__file__).resolve().parents[1]
A=ROOT/'artifacts'


def read(name):return json.loads((A/name).read_text(encoding='utf-8'))
def block(obj):return '```json\n'+json.dumps(obj,indent=2,ensure_ascii=False)+'\n```'
def table(metrics,columns):
    rows=['|模型|'+'|'.join(columns)+'|','|---|'+'|'.join(['---:']*len(columns))+'|']
    for name,result in metrics.items():
        m=result['test']
        if columns[0] not in m:continue
        rows.append('|'+name+'|'+'|'.join(f'{m[k]:.6f}' if isinstance(m[k],float) else str(m[k]) for k in columns)+'|')
    return '\n'.join(rows)


def main():
    metrics=read('metrics.json');eda=read('eda.json');config=read('config.json');env=read('environment.json');summary=read('training_summary.json')
    ranks=table(metrics,['Recall@10','HitRate@10','NDCG@10','Recall@20','HitRate@20','NDCG@20','users'])
    coverage=table(metrics,['Coverage@20','MeanPopularity@20'])
    classification=table(metrics,['AUC','LogLoss','samples','positive_rate'])
    best=max((n for n,m in metrics.items() if 'NDCG@20' in m['test']),key=lambda n:metrics[n]['test']['NDCG@20'])
    tt=metrics['TwoTower']['test']['NDCG@20'];end=metrics['TwoTower_FAISS_DeepFM']['test']['NDCG@20'];unmasked=metrics['TwoTower_no_positive_mask']['test']['NDCG@20']
    interpretation=f"本次测试 NDCG@20 最高的是 **{best}**。主双塔 NDCG@20={tt:.6f}，未屏蔽训练正例的消融={unmasked:.6f}；端到端 FAISS+DeepFM={end:.6f}，相对双塔绝对差={end-tt:+.6f}。这些是单 seed 实测，不能外推成普遍提升。若精排退化，首先检查均匀负例与召回候选分布偏移，而不是用较高 AUC 替代端到端评价。"
    results='### 测试集全目录/端到端排名\n\n'+ranks+'\n\n### 覆盖与热度\n\n'+coverage+'\n\n### 采样负例分类指标（非 CTR）\n\n'+classification
    substitutions={'EDA':block(eda),'CONFIG':block(config),'ENV':block(env),'SUMMARY':block(summary),'RESULTS':results,'INTERPRETATION':interpretation,'QA':(ROOT/'docs/INTERVIEW_QA.md').read_text(encoding='utf-8')}
    report=(ROOT/'docs/REPORT_TEMPLATE.md').read_text(encoding='utf-8')
    for k,v in substitutions.items():report=report.replace('{{'+k+'}}',v)
    assert '{{' not in report
    (ROOT/'docs/TECHNICAL_REPORT.md').write_text(report,encoding='utf-8')
    req='\n'.join(f'{p}=={v}' for p,v in env['packages'].items())+'\n'
    (ROOT/'requirements.txt').write_text(req,encoding='utf-8')
    readme=f'''# BookRec · Goodreads 图书推荐系统

真实 Goodreads 诗歌数据、CPU 实验、PyTorch 双塔 InfoNCE、FAISS Top-K、DeepFM、FastAPI 与网页展示。完整技术报告包含 60 个面试追问、数学推导、所有超参数、实际结果与局限。

## 本次运行

- 原始评论：{eda['raw_reviews']:,}；训练正反馈：{eda['train_interactions']:,}。
- 训练目录：{eda['users']:,} 用户 / {eda['items']:,} 图书；评分 ≥4，训练 5-Core。
- 全局时间切分，warm-only 排名评估；冷启动过滤数见技术报告。
- Python {env['python']} / CPU / seed {config['seed']}。

{ranks}

{classification}

{interpretation}

## 运行

建议 Python 3.13。首次运行下载官方数据、安装依赖并训练，需网络。

```powershell
powershell -ExecutionPolicy Bypass -File .\\run.ps1 -Serve
```

```bash
bash run.sh --serve
```

已有环境和模型时，直接启动服务：

```bash
python -m uvicorn bookrec.api:app --host 127.0.0.1 --port 8000
```

浏览器打开 http://127.0.0.1:8000；API 示例 `/recommend?user_id=0&k=10`；交互接口文档 `/docs`。user_id 为内部编号，合法范围 0–{eda['users']-1}，更大编号走热门兜底。

手动复现：

```bash
python -m pip install -r requirements.txt
python -m bookrec.pipeline
python -m pytest -q
python scripts/build_report.py
```

## 项目结构与证据

- [技术报告](docs/TECHNICAL_REPORT.md)：数据、EDA、架构、公式、训练、实验、风险、A/B 方案、60 问。
- [配置](configs/default.json) / [依赖](requirements.txt) / [测试](tests/)。
- [真实指标](artifacts/metrics.json) / [对比 CSV](artifacts/comparison.csv) / [训练日志](artifacts/training_log.csv)。
- [训练耗时与选模](artifacts/training_summary.json) / [环境](artifacts/environment.json) / [源文件指纹](artifacts/data_manifest.json)。
- 本地 `artifacts/` 包含 checkpoint、FAISS、向量、固定精排样本；Git 忽略模型与原始/处理数据，克隆后需重建。
- 每次运行覆盖结果；多实验请先保存当前 artifacts，报告由实际结果自动生成。

![Loss](artifacts/loss_curves.png)
![EDA](artifacts/eda.png)

## 诚实边界与后续迭代

FAISS 使用精确 FlatIP，并非 ANN；AUC/LogLoss 为采样负例协议，不是实际点击率；没有线上 A/B 或生产部署。warm-only、诗歌体裁、单 seed 和 ID-only 是主要局限。优先做训练内难负例、内容特征、多 seed/时间回测，再扩展 ANN 与线上实验。

数据来自 [UCSD Book Graph](https://mcauleylab.ucsd.edu/public_datasets/gdrive/goodreads/UCSD%20Book%20Graph.html)，使用条件由数据源决定，项目代码许可不覆盖源数据。
'''
    (ROOT/'README.md').write_text(readme,encoding='utf-8')
    print('Generated README.md, TECHNICAL_REPORT.md, requirements.txt from actual artifacts')


if __name__=='__main__':main()
