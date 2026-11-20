"""Check whether an initial broad search pays off on held-out comparison questions."""

from __future__ import annotations

import argparse
import hashlib
import json
import re
from collections import Counter, defaultdict
from pathlib import Path
from statistics import mean

from experiment import LATEST, TOKEN, TOP_K, Index, Question, build, evaluate, run, save_csv, select


CALIBRATION_SEEDS = (20261121, 20261122, 20261123)
TEST_SEEDS = (20261124, 20261125, 20261126, 20261127, 20261128, 20261129)
SCENARIOS = ("official_both_named", "mixed_both_named", "mixed_second_omitted")
POLICIES = ("all_targeted", "gap", "calibrated")


def broad_query(question: Question, omit_second: bool) -> str:
    if not omit_second:
        return question.text
    second_entity = question.slots[1][0]
    return re.sub(rf"\b{re.escape(second_entity)}\b", "the other tool", question.text, count=1)


def query_shape(question: Question, query: str) -> str:
    tokens = set(TOKEN.findall(query.lower()))
    return "both_named" if all(entity in tokens for entity, _ in question.slots) else "second_omitted"


def choose_policy(expected_hits: float, broad_cost: float = 1, scoped_cost: float = 1) -> str:
    """A broad call must offset more than its own cost in avoided scoped calls."""
    return "gap" if expected_hits * scoped_cost > broad_cost else "all_targeted"


def cases(seeds: tuple[int, ...]):
    for seed in seeds:
        official, noise, questions, gold = build(seed)
        views = {"official": Index(official), "mixed": Index(official + noise)}
        conditions = (("official", False), ("mixed", False), ("mixed", True))
        current = {q.slots[0][0]: LATEST for q in questions if q.intent == "navigate"}
        for question in questions:
            if question.intent != "compare":
                continue
            for view, omit_second in conditions:
                query = broad_query(question, omit_second)
                shape = f"{view}_{query_shape(question, query)}"
                index = views[view]
                first = index.search(query, TOP_K)
                checked = [select(first, slot, current, strict=True) for slot in question.slots]
                assert not any(conflict for _, conflict in checked)
                hits = sum(passage is not None for passage, _ in checked)
                expected = tuple(gold[slot] for slot in question.slots)
                yield seed, question, query, shape, hits, first, index, current, expected


def summarize(rows: list[dict]) -> dict:
    counts = Counter(row["status"] for row in rows)
    calls = sum(row["calls"] for row in rows)
    return {
        "questions": len(rows),
        "first_round_usable_mean": round(mean(row["first_round_usable"] for row in rows), 4),
        "grounded": counts["grounded"],
        "wrong": counts["wrong_answer"],
        "missed": counts["missed_answer"],
        "calls": calls,
        "mean_calls": round(calls / len(rows), 4),
        "mean_passages": round(mean(row["passages"] for row in rows), 4),
    }


def replay(out: Path) -> None:
    assert set(CALIBRATION_SEEDS).isdisjoint(TEST_SEEDS)
    assert choose_policy(1) == "all_targeted"
    assert choose_policy(1.1) == "gap"
    assert choose_policy(1.1, broad_cost=2) == "all_targeted"
    calibration = defaultdict(list)
    for _, _, _, shape, hits, *_ in cases(CALIBRATION_SEEDS):
        calibration[shape].append(hits)
    estimates = {shape: mean(calibration[shape]) for shape in SCENARIOS}
    choices = {shape: choose_policy(estimates[shape]) for shape in SCENARIOS}

    records = []
    groups = defaultdict(list)
    seed_groups = defaultdict(list)
    for seed, q, query, shape, hits, first, index, current, expected in cases(TEST_SEEDS):
        results = {
            "all_targeted": run("all_targeted", q, index, current, [], []),
            "gap": run("gap", q, index, current, first, []),
        }
        assert results["gap"].calls == 1 + len(q.slots) - hits
        assert results["all_targeted"].calls == len(q.slots)
        for policy in POLICIES:
            branch = choices[shape] if policy == "calibrated" else policy
            outcome = results[branch]
            row = {
                "scenario": shape, "seed": seed, "question": q.id, "broad_query": query,
                "policy": policy, "branch": branch, "first_round_usable": hits,
                "status": evaluate(outcome, expected), "calls": outcome.calls,
                "passages": outcome.returned, "citations": "|".join(outcome.citations),
            }
            records.append(row)
            groups[shape, policy].append(row)
            seed_groups[shape, seed, policy].append(row)

    summary = [{"scenario": shape, "policy": policy, **summarize(groups[shape, policy])}
               for shape in SCENARIOS for policy in POLICIES]
    by_seed = [{"scenario": shape, "seed": seed, "policy": policy,
                **summarize(seed_groups[shape, seed, policy])}
               for shape in SCENARIOS for seed in TEST_SEEDS for policy in POLICIES]
    out.mkdir(parents=True, exist_ok=True)
    save_csv(out / "summary.csv", summary)
    save_csv(out / "by_seed.csv", by_seed)
    save_csv(out / "by_query.csv", records)
    metadata = {
        "calibration_seeds": CALIBRATION_SEEDS, "test_seeds": TEST_SEEDS,
        "top_k": TOP_K, "broad_cost": 1, "scoped_cost": 1,
        "scenario": {
            "official_both_named": "current official passages; original broad query",
            "mixed_both_named": "official current, archived and mirrored passages; original broad query",
            "mixed_second_omitted": "same mixed index; second entity omitted from broad query only",
        },
        "calibration": {
            shape: {"n": len(calibration[shape]), "mean_usable": round(estimates[shape], 4),
                    "chosen_branch": choices[shape]} for shape in SCENARIOS
        },
        "sha256": {name: hashlib.sha256((out / name).read_bytes()).hexdigest()
                   for name in ("summary.csv", "by_seed.csv", "by_query.csv")},
    }
    (out / "run.json").write_text(json.dumps(metadata, ensure_ascii=False, indent=2) + "\n",
                                 encoding="utf-8", newline="\n")
    print(f"{len(TEST_SEEDS)} held-out seeds; {len(records)} policy results.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", type=Path, default=Path("results/routing"))
    replay(parser.parse_args().out)
