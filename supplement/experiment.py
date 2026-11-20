#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""RAG 答案血缘的最小充分集 —— 可复现实证脚本

问题：同一条问题、同一份语料，重建一次索引之后答案可能变了。
     要让一次回答能被原样重放或复核，trace 里到底必须钉住哪些字段？

设计要点
  1. 检索器为确定性原型：字符 2/3-gram 有符号哈希嵌入 + 余弦相似度，
     无外部模型、无 GPU、无随机数，任何一次运行都可逐字节复核。
  2. 读者在 top-k 中取“第一条同时含站点号与该字段”的片段作答，
     因此陈旧副本一旦排在明细之前，就会直接答出旧值。
  3. 六个单因子构型变更（D1–D6）分别对应线上真实发生过的操作，
     再用 L0–L4 五级累计钉住、逐字段消融与 2×2 平局实验拆开归因。

用法：
    python experiment.py --out results
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import random
import re
import struct
import unicodedata
from pathlib import Path

DIM = 8192          # 哈希嵌入维度
TOPK = 5            # 交给读者的片段数
N_DOC = 120         # 站点档案数
N_QUERY = 120       # 问句数
STALE = 20          # 汇总表中保留旧值的站点数
SEED = 20261120

REGIONS = ["东海北部", "南海西部", "黄海中部", "渤海湾", "西太平洋"]
UNITS = ["青岛海洋站", "厦门海洋站", "三亚海洋站", "舟山海洋站", "大连海洋站"]
ATTRS = [
    ("电池组更换周期", "天", 60, 120),
    ("传感器校准间隔", "个月", 3, 12),
    ("采样频率", "分钟", 5, 30),
    ("最大布放深度", "米", 50, 400),
    ("通信窗口时长", "分钟", 10, 45),
    ("维护巡检间隔", "周", 2, 8),
]
PREFIX = ["B", "M", "A"]
MODELS = ["SZ-3 型潜标", "HY-2 型浮标", "AQ-5 型剖面浮标", "LW-1 型锚系潜标"]


def station_ids() -> list[str]:
    ids = []
    for p in PREFIX:
        for i in range(1, N_DOC // len(PREFIX) + 1):
            ids.append(f"{p}{i:02d}")
    return ids[:N_DOC]


# --------------------------------------------------------------------------
# 1. 语料：海洋观测站运维档案
#    刻意保留真实语料的两个特征：同构模板造成的近重复，
#    以及“明细档案 + 汇总表”造成的同事实多副本（汇总表留着旧值）。
# --------------------------------------------------------------------------

def build_corpus(rng: random.Random, with_summary: bool = False):
    docs, gold = [], {}
    for sid in station_ids():
        region, unit, model = rng.choice(REGIONS), rng.choice(UNITS), rng.choice(MODELS)
        rev, year = rng.randint(2, 9), rng.randint(2019, 2025)
        month = rng.randint(1, 12)
        vals, lines = {}, []
        lines.append(f"站点 {sid} 观测设备运维档案（第 {rev} 版）")
        lines.append(
            f"本档案记录 {sid} 号观测站在{region}的常规运维参数，由{unit}负责执行与复核。"
            f"现役设备为{model}，{year} 年 {month} 月完成布放，最近一次改版依据年度巡检结论。")
        lines.append(
            f"历史沿革：该站自建成以来共经历 {rng.randint(2, 6)} 次设备更换，"
            f"早期使用的传感器已停产，备件由{unit}统一保管。")
        for idx, (attr, unit_, lo, hi) in enumerate(ATTRS, start=1):
            v = rng.randint(lo, hi)
            vals[attr] = v
            tail = rng.choice([
                f"该值在上一周期未作调整，若遇连续阴雨需提前执行。",
                f"执行时以{unit}现场记录表为准，异常情况需当日上报。",
                f"该参数与同类站点的偏差在允许范围内，无需单独说明。",
                f"调整需经值班长复核，并在配置库中登记变更原因。",
            ])
            lines.append(f"{idx}. {attr}为 {v} {unit_}。{tail}")
        lines.append(
            f"异常记录：上一年度共发生 {rng.randint(0, 4)} 次通信中断，"
            f"均在 {rng.randint(2, 12)} 小时内恢复，未影响数据完整性。")
        lines.append(f"备注：以上参数由{unit}归档，历史版本保留在配置库中。")
        docs.append({"doc_id": f"DOC-{sid}", "sid": sid, "text": "\n".join(lines)})
        gold[sid] = vals

    if not with_summary:
        return docs, gold
    lines = ["各站电池组更换周期汇总（2025 年 12 月版）",
             "本表供值班人员快速比对，明细以各站档案为准，如有出入以档案为准。"]
    for sid in station_ids()[:STALE]:
        old = gold[sid]["电池组更换周期"] + rng.choice([-9, -7, 7, 9])
        lines.append(f"站点 {sid} 的电池组更换周期为 {old} 天。")
    docs.append({"doc_id": "DOC-SUM", "sid": "SUM", "text": "\n".join(lines)})
    return docs, gold


def make_queries(gold, rng: random.Random):
    sids = station_ids()
    qs = []
    for i in range(N_QUERY):
        sid = sids[(i * 7) % N_DOC]
        attr, unit_, _, _ = ATTRS[i % len(ATTRS)]
        qs.append({"qid": f"Q{i + 1:02d}", "sid": sid, "attr": attr,
                   "query": f"站点 {sid} 的{attr}是多少{unit_}？",
                   "gold": gold[sid][attr]})
    return qs


# --------------------------------------------------------------------------
# 2. 文本变体与归一化
# --------------------------------------------------------------------------

def reflow(text: str) -> str:
    """模拟 PDF/HTML 重新抽取：按 24 字硬折行、标点后补空格、段间插空行。"""
    out = []
    for line in text.split("\n"):
        line = line.replace("。", "。 ").replace("，", "， ").replace("、", "、 ")
        while len(line) > 24:
            out.append(line[:24])
            line = line[24:]
        out.append(line)
        out.append("")
    return "\n".join(out)


def apply_updates(docs, targets, rng: random.Random):
    """targets: {(站点, 字段)} —— 按字段逐一改写，模拟一次真实的档案改版。"""
    by_sid = {}
    for sid, attr in targets:
        by_sid.setdefault(sid, []).append(attr)
    out = []
    for d in docs:
        text = d["text"]
        for attr in by_sid.get(d["sid"], []):
            m = re.search(rf"{attr}为 (\d+) ", text)
            if m:
                new = int(m.group(1)) + rng.choice([-11, -5, 5, 11])
                text = text.replace(m.group(0), f"{attr}为 {new} ", 1)
        out.append({**d, "text": text})
    return out


def canon_v1(text: str) -> str:
    """归档时的归一化实现：逐行去首尾空白、行内空白折叠、丢空行。"""
    lines = [re.sub(r"[ \t]+", " ", ln).strip() for ln in text.split("\n")]
    return "\n".join([ln for ln in lines if ln])


def canon_v2(text: str) -> str:
    """升级后的归一化实现：NFKC + 去尽空白。

    中文正文里空白不承载语义，仓储层的规范化实现通常会把空白整体去掉；
    这样同一个语义文本无论被抽取多少次（硬折行、标点后补空格、段间空行），
    都会得到同一个字符流。代价是混排英文的子词边界会被抹平，列为局限。
    """
    t = unicodedata.normalize("NFKC", text)
    return re.sub(r"\s+", "", t)


CANON = {"v1": canon_v1, "v2": canon_v2}


# --------------------------------------------------------------------------
# 3. 分块、嵌入、索引
# --------------------------------------------------------------------------

def chunk_doc(doc, canon_fn, width: int, overlap: int, cid_mode: str):
    body = canon_fn(doc["text"])
    header = f"[{doc['doc_id']} 站点 {doc['sid']}] "
    step = max(1, width - overlap)
    out = []
    for i, start in enumerate(range(0, max(1, len(body) - overlap), step)):
        seg = body[start:start + width]
        if not seg.strip():
            break
        text = header + seg
        if cid_mode == "content":
            cid = f"{doc['doc_id']}#" + hashlib.sha256(text.encode()).hexdigest()[:10]
        else:
            cid = f"{doc['doc_id']}#{i:03d}"
        out.append({"cid": cid, "doc_id": doc["doc_id"], "sid": doc["sid"],
                    "text": text, "span": (start, start + len(seg))})
    return out


def _f32(x: float) -> float:
    return struct.unpack("f", struct.pack("f", x))[0]


def embed(text: str, dtype: str = "f64") -> dict[int, float]:
    """字符 2-gram + 3-gram 有符号哈希嵌入（确定性，无需外部模型）。"""
    vec: dict[int, float] = {}
    t = re.sub(r"\s+", "", text)
    grams = [t[i:i + 2] for i in range(len(t) - 1)]
    grams += [t[i:i + 3] for i in range(len(t) - 2)]
    for g in grams:
        h = int.from_bytes(hashlib.blake2b(g.encode("utf-8"), digest_size=8).digest(), "big")
        idx = h % DIM
        vec[idx] = vec.get(idx, 0.0) + (1.0 if (h >> 63) & 1 == 0 else -1.0)
    if dtype == "f32":
        vec = {k: _f32(v) for k, v in vec.items()}
    norm = math.sqrt(sum(v * v for v in vec.values())) or 1.0
    return {k: v / norm for k, v in vec.items()}


class Index:
    def __init__(self, chunks, dtype, order_seed, tie):
        self.tie = tie
        self.chunks = list(chunks)
        if order_seed is not None:
            random.Random(order_seed).shuffle(self.chunks)
        self.vecs = [embed(c["text"], dtype) for c in self.chunks]

    def search(self, qvec, k=TOPK):
        scored = []
        for pos, (c, v) in enumerate(zip(self.chunks, self.vecs)):
            s = 0.0
            for d, w in qvec.items():
                cw = v.get(d)
                if cw is not None:
                    s += w * cw
            key = pos if self.tie == "insertion" else c["cid"]
            scored.append((s, key, c))
        scored.sort(key=lambda x: (-x[0], x[1]))
        return [(s, c) for s, _, c in scored[:k]], scored


# --------------------------------------------------------------------------
# 4. 读者与答案抽取
# --------------------------------------------------------------------------

def extract(chunk_text: str, sid: str, attr: str):
    if f"站点 {sid}" not in chunk_text:
        return None
    m = re.search(rf"{attr}为\s*([0-9]+(?:\.[0-9]+)?)", chunk_text)
    return m.group(1) if m else None


def answer(qvec, index, q):
    hits, _ = index.search(qvec)
    for score, c in hits:
        val = extract(c["text"], q["sid"], q["attr"])
        if val is not None:
            return {"value": val, "doc_id": c["doc_id"], "cid": c["cid"],
                    "rank": [h[1]["cid"] for h in hits].index(c["cid"]) + 1,
                    "docs": [h[1]["doc_id"] for h in hits],
                    "chunks": [h[1]["cid"] for h in hits],
                    "top_score": round(hits[0][0], 6)}
    top = hits[0][1]
    return {"value": "未找到", "doc_id": top["doc_id"], "cid": top["cid"],
            "rank": 0, "docs": [h[1]["doc_id"] for h in hits],
            "chunks": [h[1]["cid"] for h in hits], "top_score": round(hits[0][0], 6)}


def count_ties(scored):
    """统计答案边界处的精确/近似平局：|s_k - s_{k+1}| 的分布。"""
    s = [x[0] for x in scored]
    if len(s) <= TOPK:
        return 0, 0
    a, b = s[TOPK - 1], s[TOPK]
    if a <= 0:
        return 0, 0
    return (1 if a == b else 0), (1 if abs(a - b) < 1e-6 else 0)


# --------------------------------------------------------------------------
# 5. 构型与度量
# --------------------------------------------------------------------------

BASE = dict(text="orig", canon="v1", width=200, overlap=80,
            order_seed=None, dtype="f64", tie="insertion", cid_mode="pos")
TODAY = dict(text="reflow_updated", canon="v2", width=240, overlap=80,
             order_seed=SEED + 7, dtype="f32", tie="insertion", cid_mode="pos")


def cfg(**kw):
    c = dict(BASE)
    c.update(kw)
    return c


def evaluate(run_ans, base_ans, queries):
    n = len(queries)
    eq = sum(run_ans[q["qid"]]["value"] == base_ans[q["qid"]]["value"] and
             run_ans[q["qid"]]["doc_id"] == base_ans[q["qid"]]["doc_id"] for q in queries)
    strict = sum(run_ans[q["qid"]]["value"] == base_ans[q["qid"]]["value"] and
                 run_ans[q["qid"]]["cid"] == base_ans[q["qid"]]["cid"] for q in queries)
    jac = sum(len(set(run_ans[q["qid"]]["docs"]) & set(base_ans[q["qid"]]["docs"])) /
              len(set(run_ans[q["qid"]]["docs"]) | set(base_ans[q["qid"]]["docs"]))
              for q in queries) / n
    acc = sum(run_ans[q["qid"]]["value"] == str(q["gold"]) for q in queries) / n
    cite_only = sum(run_ans[q["qid"]]["value"] == base_ans[q["qid"]]["value"] and
                    run_ans[q["qid"]]["cid"] != base_ans[q["qid"]]["cid"] for q in queries)
    chg = sum(run_ans[q["qid"]]["value"] != base_ans[q["qid"]]["value"] or
              run_ans[q["qid"]]["doc_id"] != base_ans[q["qid"]]["doc_id"] for q in queries)
    return {"answer_equal": eq / n, "strict_replay": strict / n, "jaccard": jac,
            "accuracy": acc, "changed": chg, "cite_only": cite_only}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="results")
    args = ap.parse_args()
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)

    rng = random.Random(SEED)
    docs, gold = build_corpus(rng)
    queries = make_queries(gold, rng)
    targets = {(q["sid"], q["attr"]) for q in queries[:STALE]}
    upd_docs = apply_updates(docs, targets, random.Random(SEED + 1))
    ref_docs = [{**d, "text": reflow(d["text"])} for d in docs]
    ref_upd = [{**d, "text": reflow(d["text"])} for d in upd_docs]
    variants = {"orig": docs, "reflow": ref_docs,
                "updated": upd_docs, "reflow_updated": ref_upd}

    qvecs = {q["qid"]: embed(q["query"]) for q in queries}

    def run(config):
        chunks = []
        for d in variants[config["text"]]:
            chunks.extend(chunk_doc(d, CANON[config["canon"]], config["width"],
                                    config["overlap"], config["cid_mode"]))
        idx = Index(chunks, config["dtype"], config["order_seed"], config["tie"])
        ans = {q["qid"]: answer(qvecs[q["qid"]], idx, q) for q in queries}
        return ans, chunks, idx

    base, base_chunks, _ = run(cfg())

    # ---- 5.1 单因子漂移 ----
    sources = [
        ("D1 重新抽取（硬折行+标点空格）", cfg(text="reflow")),
        ("D2 归一化升级 v1→v2", cfg(canon="v2")),
        ("D3 分块参数 200/80→240/80", cfg(width=240, overlap=80)),
        ("D4 索引构建顺序打乱", cfg(order_seed=SEED + 7)),
        ("D5 数值精度 f64→f32", cfg(dtype="f32")),
        ("D6 真实内容更新（20 站）", cfg(text="updated")),
    ]
    rows_src = []
    for name, c in sources:
        a, _, _ = run(c)
        m = evaluate(a, base, queries)
        rows_src.append({"drift_source": name, "answer_equal": round(m["answer_equal"], 4),
                         "jaccard": round(m["jaccard"], 4),
                         "strict_replay": round(m["strict_replay"], 4),
                         "same_answer_new_citation": m["cite_only"],
                         "changed_queries": m["changed"]})

    # ---- 5.2 累计钉住 ----
    levels = [
        ("L0 只存问题与答案", dict(TODAY)),
        ("L1 +钉语料快照", {**TODAY, "text": "orig"}),
        ("L2 +钉归一化版本", {**TODAY, "text": "orig", "canon": "v1"}),
        ("L3 +钉分块参数", {**TODAY, "text": "orig", "canon": "v1",
                         "width": 200, "overlap": 80}),
        ("L4 +钉构建顺序与数值精度", dict(BASE)),
    ]
    rows_pin = []
    for name, c in levels:
        a, _, _ = run(c)
        m = evaluate(a, base, queries)
        rows_pin.append({"pinning": name, "answer_equal": round(m["answer_equal"], 4),
                         "strict_replay": round(m["strict_replay"], 4),
                         "jaccard": round(m["jaccard"], 4),
                         "accuracy": round(m["accuracy"], 4),
                         "same_answer_new_citation": m["cite_only"],
                         "changed_queries": m["changed"]})

    # ---- 5.3 逐字段消融：其余字段钉住，只放开这一个 ----
    released = [
        ("语料快照", {"text": "reflow_updated"}),
        ("归一化版本", {"canon": "v2"}),
        ("分块参数", {"width": 240, "overlap": 80}),
        ("构建顺序", {"order_seed": SEED + 7}),
        ("数值精度", {"dtype": "f32"}),
        ("平局规则", {"tie": "cid"}),
    ]
    rows_abl = []
    for name, kw in released:
        c = cfg(**kw)
        a, _, _ = run(c)
        m = evaluate(a, base, queries)
        rows_abl.append({"field_released": name,
                         "answer_equal": round(m["answer_equal"], 4),
                         "fail_rate": round(1 - m["answer_equal"], 4),
                         "strict_replay": round(m["strict_replay"], 4)})

    # ---- 5.4 平局 × 顺序 2×2，并记录平局频次 ----
    cells, tie_counts = [], []
    for tie in ("insertion", "cid"):
        for order in (None, SEED + 7):
            c = cfg(tie=tie, order_seed=order,
                    cid_mode="content" if tie == "cid" else "pos")
            a, _, idx = run(c)
            b, _, _ = run(cfg(tie=tie, cid_mode="content" if tie == "cid" else "pos"))
            m = evaluate(a, b, queries)
            ex = nr = 0
            for q in queries:
                _, scored = idx.search(qvecs[q["qid"]])
                e, n2 = count_ties(scored)
                ex += e
                nr += n2
            cells.append({"tie_rule": tie,
                          "order": "fixed" if order is None else "shuffled",
                          "answer_equal": round(m["answer_equal"], 4),
                          "strict_replay": round(m["strict_replay"], 4)})
            tie_counts.append({"tie_rule": tie,
                               "order": "fixed" if order is None else "shuffled",
                               "exact_ties": ex, "near_ties": nr})

    # ---- 5.5 硬化管道：归一化 v2 + 内容寻址 id + 按 id 定序 ----
    hard = cfg(text="reflow", canon="v2", cid_mode="content", tie="cid")
    hard_ref = cfg(text="orig", canon="v2", cid_mode="content", tie="cid")
    a, _, _ = run(hard)
    b, _, _ = run(hard_ref)
    m_hard = evaluate(a, b, queries)
    plain, _, _ = run(cfg(text="reflow"))
    m_plain = evaluate(plain, base, queries)
    # 硬化与归档的差异只应来自 id 方案：比较 (value, doc) 与 (value, cid)
    cross = evaluate(a, base, queries)

    # ---- 5.6 变更检测 ----
    today_all, _, _ = run(dict(TODAY))
    today_pinned, _, _ = run(cfg(text="updated"))

    def detect(cur):
        declared = {q["qid"] for q in queries
                    if cur[q["qid"]]["value"] != base[q["qid"]]["value"]}
        truth = {q["qid"] for q in queries if (q["sid"], q["attr"]) in targets}
        tp = len(declared & truth)
        return {"declared": len(declared), "true": len(truth),
                "precision": round(tp / len(declared), 4) if declared else 0.0,
                "recall": round(tp / len(truth), 4) if truth else 0.0}

    det = [{"mode": "无血缘（今日全构型 vs 归档）", **detect(today_all)},
           {"mode": "钉住构型（归档构型 + 今日语料）", **detect(today_pinned)}]

    # ---- 5.7 旁证：加入一份“留着旧值的汇总表”会怎样 ----
    # 同事实多副本在真实语料里极常见。它不只制造错答，还会把真实更新遮蔽掉：
    # 读者命中陈旧副本，明细改了也看不出来。
    dsum, gsum = build_corpus(random.Random(SEED), with_summary=True)
    sdoc = [d for d in dsum if d["doc_id"] == "DOC-SUM"][0]
    sdoc_upd = {**sdoc, "text": reflow(sdoc["text"])}
    upd_map = {d["doc_id"]: d for d in upd_docs}

    def run_sum(summary_text, detail_docs):
        chunks = []
        for d in list(detail_docs) + [{"doc_id": "DOC-SUM", "sid": "SUM", "text": summary_text}]:
            chunks.extend(chunk_doc(d, canon_v1, 200, 80, "pos"))
        idx = Index(chunks, "f64", None, "insertion")
        return {q["qid"]: answer(qvecs[q["qid"]], idx, q) for q in queries}

    s_orig = run_sum(sdoc["text"], docs)
    s_upd = run_sum(sdoc["text"], [upd_map[d["doc_id"]] for d in docs])
    shadow_rows = [{
        "condition": "含陈旧汇总表（未更新）",
        "answered_from_summary": sum(1 for q in queries
                                     if s_orig[q["qid"]]["doc_id"] == "DOC-SUM"),
        "accuracy": round(sum(s_orig[q["qid"]]["value"] == str(q["gold"])
                              for q in queries) / len(queries), 4),
        "true_updates_detected": sum(1 for q in queries if (q["sid"], q["attr"]) in targets
                                     and s_upd[q["qid"]]["value"] != s_orig[q["qid"]]["value"]),
        "true_updates_total": len([q for q in queries if (q["sid"], q["attr"]) in targets]),
    }, {
        "condition": "无陈旧副本（主实验）",
        "answered_from_summary": 0,
        "accuracy": round(sum(base[q["qid"]]["value"] == str(q["gold"])
                              for q in queries) / len(queries), 4),
        "true_updates_detected": sum(1 for q in queries if (q["sid"], q["attr"]) in targets
                                     and today_pinned[q["qid"]]["value"] != base[q["qid"]]["value"]),
        "true_updates_total": len([q for q in queries if (q["sid"], q["attr"]) in targets]),
    }]

    # ---- 写出 ----
    def dump(name, rows):
        with open(out / name, "w", newline="", encoding="utf-8-sig") as f:
            w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
            w.writeheader()
            w.writerows(rows)

    dump("summary.csv", rows_pin)
    dump("sources.csv", rows_src)
    dump("ablation.csv", rows_abl)
    dump("tie_order.csv", cells)
    dump("tie_counts.csv", tie_counts)
    dump("change_detection.csv", det)
    dump("stale_shadow.csv", shadow_rows)

    with open(out / "by_query.csv", "w", newline="", encoding="utf-8-sig") as f:
        w = csv.writer(f)
        w.writerow(["qid", "station", "attr", "gold", "base_value", "base_doc", "base_rank",
                    "L0_value", "L0_doc", "L4_value", "L4_doc"])
        for q in queries:
            w.writerow([q["qid"], q["sid"], q["attr"], q["gold"],
                        base[q["qid"]]["value"], base[q["qid"]]["doc_id"],
                        base[q["qid"]]["rank"],
                        today_all[q["qid"]]["value"], today_all[q["qid"]]["doc_id"],
                        today_pinned[q["qid"]]["value"], today_pinned[q["qid"]]["doc_id"]])

    payload = {
        "seed": SEED, "dim": DIM, "topk": TOPK, "docs": len(docs),
        "queries": len(queries), "chunks_base": len(base_chunks),
        "base_accuracy": round(sum(base[q["qid"]]["value"] == str(q["gold"])
                                   for q in queries) / len(queries), 4),
        "base_notfound": sum(base[q["qid"]]["value"] == "未找到" for q in queries),
        "hardened_reflow_vs_orig": {k: round(v, 4) if isinstance(v, float) else v
                                    for k, v in m_hard.items()},
        "hardened_vs_archive": {k: round(v, 4) if isinstance(v, float) else v
                                for k, v in cross.items()},
        "plain_reflow_vs_archive": {k: round(v, 4) if isinstance(v, float) else v
                                    for k, v in m_plain.items()},
        "stale_shadow": shadow_rows,
        "sha256": {},
    }
    for fn in ("summary.csv", "sources.csv", "ablation.csv", "tie_order.csv",
               "tie_counts.csv", "change_detection.csv", "stale_shadow.csv"):
        payload["sha256"][fn] = hashlib.sha256((out / fn).read_bytes()).hexdigest()[:16]
    (out / "run.json").write_text(json.dumps(payload, ensure_ascii=False, indent=2),
                                  encoding="utf-8")

    svg_pinning(out / "fig_pinning.svg", rows_pin)
    svg_sources(out / "fig_sources.svg", rows_src)

    print(json.dumps(payload, ensure_ascii=False, indent=2))
    for r in rows_pin + rows_src + rows_abl + cells + tie_counts + det + shadow_rows:
        print(r)


def svg_pinning(path, rows):
    W, H, pad = 600, 320, 60
    xs = [pad + i * (W - 2 * pad) / (len(rows) - 1) for i in range(len(rows))]

    def y(v):
        return H - pad - v * (H - 2 * pad)

    pts = " ".join(f"{x:.1f},{y(r['answer_equal']):.1f}" for x, r in zip(xs, rows))
    dots = "".join(f'<circle cx="{x:.1f}" cy="{y(r["answer_equal"]):.1f}" r="4.5" fill="#1a6fb5"/>'
                   for x, r in zip(xs, rows))
    labels = "".join(
        f'<text x="{x:.1f}" y="{H - pad + 20}" font-size="11" fill="#555" '
        f'text-anchor="middle">{r["pinning"].split()[0]}</text>'
        f'<text x="{x:.1f}" y="{y(r["answer_equal"]) - 10:.1f}" font-size="10" '
        f'fill="#1a6fb5" text-anchor="middle">{r["answer_equal"] * 100:.0f}%</text>'
        for x, r in zip(xs, rows))
    svg = f'''<svg xmlns="http://www.w3.org/2000/svg" width="{W}" height="{H}" viewBox="0 0 {W} {H}">
<rect width="{W}" height="{H}" fill="#ffffff"/>
<line x1="{pad}" y1="{H - pad}" x2="{W - pad}" y2="{H - pad}" stroke="#bbbbbb"/>
<line x1="{pad}" y1="{pad}" x2="{pad}" y2="{H - pad}" stroke="#bbbbbb"/>
<text x="{pad}" y="26" font-size="14" fill="#222">累计钉住级别与答案等价率</text>
<text x="10" y="{pad + 4}" font-size="10" fill="#555">100%</text>
<text x="16" y="{H - pad + 4}" font-size="10" fill="#555">0%</text>
<polyline points="{pts}" fill="none" stroke="#1a6fb5" stroke-width="2.2"/>{dots}{labels}
</svg>'''
    Path(path).write_text(svg, encoding="utf-8")


def svg_sources(path, rows):
    """分组条形图：答案漂移率 vs 证据churn 率。"""
    W, left, top, rh = 720, 250, 56, 46
    H = top + rh * len(rows) + 46
    bw = W - left - 90
    svg = [f'<svg xmlns="http://www.w3.org/2000/svg" width="{W}" height="{H}" viewBox="0 0 {W} {H}">',
           f'<rect width="{W}" height="{H}" fill="#ffffff"/>',
           f'<text x="{left}" y="26" font-size="14" fill="#222">单因子构型变更：答案漂移与证据churn</text>',
           f'<rect x="{left}" y="36" width="10" height="10" fill="#c0392b"/>',
           f'<text x="{left+15}" y="45" font-size="10.5" fill="#333">答案漂移率</text>',
           f'<rect x="{left+105}" y="36" width="10" height="10" fill="#2c7fb8"/>',
           f'<text x="{left+120}" y="45" font-size="10.5" fill="#333">证据churn 率（1 − top-5 Jaccard）</text>']
    for i, r in enumerate(rows):
        yy = top + i * rh
        a = 1 - r["answer_equal"]
        e = 1 - r["jaccard"]
        svg.append(f'<text x="8" y="{yy+15}" font-size="10.5" fill="#333">{r["drift_source"]}</text>')
        svg.append(f'<rect x="{left}" y="{yy}" width="{a*bw:.1f}" height="13" fill="#c0392b"/>')
        svg.append(f'<text x="{left+a*bw+5:.1f}" y="{yy+11}" font-size="9.5" fill="#c0392b">{a*100:.1f}%</text>')
        svg.append(f'<rect x="{left}" y="{yy+16}" width="{e*bw:.1f}" height="13" fill="#2c7fb8"/>')
        svg.append(f'<text x="{left+e*bw+5:.1f}" y="{yy+27}" font-size="9.5" fill="#2c7fb8">{e*100:.1f}%</text>')
    svg.append(f'<line x1="{left}" y1="{top-6}" x2="{left}" y2="{top+len(rows)*rh-8}" stroke="#bbbbbb"/>')
    svg.append(f'<text x="{left}" y="{H-14}" font-size="10" fill="#555">'
               f'纵轴：120 条问句中被改变的比例</text>')
    svg.append('</svg>')
    Path(path).write_text("\n".join(svg), encoding="utf-8")


if __name__ == "__main__":
    main()
