# 补充实验 答案出处的复原

本实验用 120 份合成运维档案和 120 条问题，比较索引重建前后的答案、出处及逐字段重放。它对应 Word 报告第 6 节，独立于主实验的 102 道问题。

- [复现脚本](experiment.py)
- [累积固定构型的重放率](results/summary.csv)
- [单因素结果](results/sources.csv)、[构型字段消融](results/ablation.csv)和[变更告警结果](results/change_detection.csv)
- [逐题结果](results/by_query.csv)及[运行记录](results/run.json)

从仓库根目录执行 `python supplement/experiment.py --out replay/supplement`。运行只需要 Python 标准库，种子为 `20261120`。结果文件的部分哈希记录在 `results/run.json`。
