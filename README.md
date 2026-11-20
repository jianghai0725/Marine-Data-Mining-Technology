# 海洋数据挖掘技术课程项目

## AI 时代下的搜索技术——以 RAG 为例

- [实验报告（Word）](AI时代下的搜索技术_实验报告.docx)
- [搜索技术演变史的完整对话](chat/搜索技术演变史_完整对话.md)
- [主实验脚本](experiment.py)、[汇总结果](results/summary.csv)、[逐题结果](results/by_query.csv)和[运行记录](results/run.json)
- [补充实验：索引重建后的答案出处复原](supplement/README.md)

主实验比较关键词排序、链接列表、证据门控与定向补搜。补充实验检验语料、归一化和分块等构型变化对既有答复及其引用能否重放的影响。两项实验分别使用合成语料和固定随机种子。

## 复现

使用 Python 3.10 或以上版本，两个脚本只依赖标准库。在仓库根目录运行：

```bash
python experiment.py --out replay/main
python supplement/experiment.py --out replay/supplement
```

将主实验和补充实验的回放结果分别与各自的 `results/run.json` 核对。主实验语料保存在 [corpus.json.gz](results/corpus.json.gz) 中。
