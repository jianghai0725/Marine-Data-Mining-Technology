"""Compare ranked links and evidence-controlled answers on synthetic manuals."""

from __future__ import annotations

import argparse
import csv
import gzip
import hashlib
import json
import math
import random
import re
from collections import Counter, defaultdict
from dataclasses import asdict, dataclass, replace
from pathlib import Path
from statistics import mean


SEED = 20261120
TOP_K = 3
WIDE_K = 7
TOKEN = re.compile(r"[a-z]+[0-9]+|[a-z]+|[0-9]+", re.I)
PROFILES = ("official_only", "mixed")
POLICIES = ("links", "fixed", "gate", "wide", "gap", "all_targeted")
LATEST = 2


@dataclass(frozen=True)
class Passage:
    id: str
    entity: str
    field: str
    value: str
    publisher: str
    version: int
    text: str


@dataclass(frozen=True)
class Question:
    id: str
    intent: str
    text: str
    slots: tuple[tuple[str, str], ...]


@dataclass(frozen=True)
class Run:
    output: str
    citations: tuple[str, ...]
    reason: str
    calls: int
    returned: int


class Index:
    def __init__(self, passages: list[Passage]):
        self.documents = {p.id: p for p in passages}
        self.tokens = {p.id: Counter(TOKEN.findall(p.text.lower())) for p in passages}
        self.avg_length = mean(counts.total() for counts in self.tokens.values())
        frequencies = Counter(token for counts in self.tokens.values() for token in counts)
        self.idf = {token: math.log1p((len(passages) - frequency + 0.5) / (frequency + 0.5))
                    for token, frequency in frequencies.items()}
        by_entity = defaultdict(list)
        for passage in passages:
            by_entity[passage.entity].append(passage.id)
        self.by_entity = dict(by_entity)

    def search(self, query: str, k: int, entity: str | None = None) -> list[Passage]:
        terms = sorted(set(TOKEN.findall(query.lower())) & self.idf.keys())
        identifiers = self.by_entity.get(entity, []) if entity else self.documents
        scored = []
        for identifier in identifiers:
            counts = self.tokens[identifier]
            normalizer = 1.2 * (0.25 + 0.75 * counts.total() / self.avg_length)
            score = sum(self.idf[term] * counts[term] * 2.2 / (counts[term] + normalizer)
                        for term in terms)
            if score > 0:
                scored.append((-score, identifier))
        return [self.documents[identifier] for _, identifier in sorted(scored)[:k]]


QUOTA_WORDS = ("export quota", "export file limit", "file export allowance")
RETENTION_WORDS = ("history retention", "history expiry", "activity records kept")
FIELD_QUERY = {"quota": "export quota file limit allowance",
               "retention": "history retention expiry activity records kept"}


def build():
    rng = random.Random(SEED)
    apps = [f"{stem}{suffix:02}" for stem in ("aster", "birch", "coral", "dune", "elm", "firth")
            for suffix in range(1, 7)]
    official, noise, questions = [], [], []
    # Gold IDs are assigned from the generated authoritative records before views are constructed.
    gold = {}
    for i, app in enumerate(apps):
        quota = f"{12 + rng.randrange(72)} files"
        days = f"{7 + rng.randrange(60)} days"
        spec = [
            ("manual", f"https://{app}.example/guide", "official manual documentation user guide"),
            ("quota", quota, rng.choice(QUOTA_WORDS)),
        ]
        if i % 3:
            spec.append(("retention", days, rng.choice(RETENTION_WORDS)))
        for field, value, words in spec:
            identifier = f"{app}/current/{field}"
            gold[app, field] = identifier
            extra = rng.choice(("", "administrator reference settings current workflow",
                                "account workspace reference table"))
            official.append(Passage(identifier, app, field, value, "official", LATEST,
                                    f"{app} {words} {value} {extra}"))
        for provenance in ("archive", "mirror"):
            for field in ("manual", "quota", "retention"):
                if provenance == "archive" and field == "manual" and i % 2:
                    continue
                value = {"manual": f"https://{app}.{provenance}.example/guide",
                         "quota": f"{rng.randrange(3, 88)} files",
                         "retention": f"{rng.randrange(2, 49)} days"}[field]
                words = {"manual": "official manual documentation user guide",
                         "quota": "export quota file limit allowance batch capacity",
                         "retention": "history retention expiry activity records kept"}[field]
                extra = "" if (i + len(field)) % 3 else "workspace guide notes feature"
                noise.append(Passage(f"{app}/{provenance}/{field}", app, field, value,
                                     "official" if provenance == "archive" else "mirror",
                                     1 if provenance == "archive" else LATEST,
                                     f"{app} {words} {value} {extra}"))
        questions.append(Question(f"{app}-nav", "navigate",
                                  f"open official current {app} manual user guide", ((app, "manual"),)))
        quota_phrase = rng.choice(QUOTA_WORDS)
        questions.append(Question(f"{app}-lookup", "lookup",
                                  f"what is the current {quota_phrase} for {app}?", ((app, "quota"),)))
        if i % 3 == 0:
            history_phrase = rng.choice(RETENTION_WORDS)
            questions.append(Question(f"{app}-missing", "missing",
                                      f"what is the current {history_phrase} for {app}?",
                                      ((app, "retention"),)))
    for i in range(0, len(apps), 2):
        first, second = apps[i:i + 2]
        phrase = rng.choice(QUOTA_WORDS)
        questions.append(Question(f"{first}-compare", "compare",
                                  f"compare the current {phrase} of {first} and {second}",
                                  ((first, "quota"), (second, "quota"))))
    return official, noise, questions, gold


def valid(passage: Passage, slot: tuple[str, str], current: dict[str, int]) -> bool:
    return ((passage.entity, passage.field) == slot
            and passage.publisher == "official"
            and passage.version == current.get(slot[0]))


def select(passages: list[Passage], slot: tuple[str, str], current: dict[str, int],
           strict: bool) -> tuple[Passage | None, bool]:
    candidates = [passage for passage in passages
                  if (passage.entity, passage.field) == slot]
    if not strict:
        return next(iter(candidates), None), False
    eligible = [p for p in candidates if valid(p, slot, current)]
    conflict = len({p.value for p in eligible}) > 1
    return (eligible[0] if eligible and not conflict else None), conflict


def run(policy: str, question: Question, index: Index,
        current: dict[str, int], first: list[Passage], wide: list[Passage]) -> Run:
    if policy == "links" or question.intent == "navigate":
        return Run("links", tuple(p.id for p in first), "ranked_links", 1, len(first))
    initial = wide if policy == "wide" else first
    if policy == "all_targeted":
        initial = []
    calls = int(policy != "all_targeted")
    returned = len(initial)
    selected = []
    for slot in question.slots:
        entity, field = slot
        passage, conflict = select(initial, slot, current, strict=policy != "fixed")
        if conflict:
            return Run("abstain", (), "conflicting_evidence", calls, returned)
        if passage is None and policy in ("gap", "all_targeted"):
            candidates = index.search(f"{entity} {FIELD_QUERY[field]}", TOP_K, entity)
            calls += 1
            returned += len(candidates)
            passage, conflict = select(candidates, slot, current, strict=True)
            if conflict:
                return Run("abstain", (), "conflicting_evidence", calls, returned)
        selected.append(passage)
    if not all(selected):
        return Run("abstain", (), "evidence_gap", calls, returned)
    return Run("answer", tuple(p.id for p in selected), "evidence_complete", calls, returned)


def evaluate(answer: Run, expected: tuple[str | None, ...]) -> str:
    has_gold = None not in expected
    if answer.output == "links":
        if not has_gold:
            return "unscored_links"
        return "covered" if all(g in answer.citations for g in expected) else "missed"
    if answer.output == "abstain":
        return "missed_answer" if has_gold else "correct_abstain"
    return "grounded" if has_gold and answer.citations == expected else "wrong_answer"


def render(out: Run, index: Index) -> str:
    if out.output == "abstain":
        return "Insufficient evidence in this snapshot."
    if out.output == "links":
        return " | ".join(index.documents[key].value if index.documents[key].field == "manual" else key
                          for key in out.citations)
    return "; ".join(f"{index.documents[key].entity}: {index.documents[key].value} [{key}]"
                     for key in out.citations)


def examples():
    question = Question("example", "compare", "compare aster01 birch01 export quota",
                        (("aster01", "quota"), ("birch01", "quota")))
    old = Passage("aster-old", "aster01", "quota", "15 files", "official", 1,
                  "aster01 export quota file limit")
    fresh_a = replace(old, id="aster-new", version=2, value="25 files")
    fresh_b = replace(fresh_a, id="birch-new", entity="birch01", value="30 files",
                      text="birch01 export quota file limit")
    index = Index([old, fresh_a, fresh_b])
    initial = [old, fresh_b]
    current = {"aster01": 2, "birch01": 2}
    expected = "aster-new", "birch-new"
    checked = {policy: run(policy, question, index, current, initial, index.search(question.text, 7))
               for policy in POLICIES}
    assert evaluate(checked["fixed"], expected) == "wrong_answer"
    assert evaluate(checked["gate"], expected) == "missed_answer"
    assert evaluate(checked["gap"], expected) == "grounded" and checked["gap"].calls == 2
    bad_manifest = {"aster01": 1, "birch01": 2}
    bad = run("gap", question, index, bad_manifest, initial, initial)
    assert evaluate(bad, expected) == "wrong_answer"
    conflict = replace(fresh_a, id="aster-conflict", value="80 files")
    rejected = run("gap", question, index, current, [fresh_a, conflict, fresh_b], initial)
    assert rejected.reason == "conflicting_evidence"
    return {**{policy: {**asdict(out), "evaluation": evaluate(out, expected),
                        "rendered": render(out, index)}
               for policy, out in checked.items()},
            "stale_release_manifest": {**asdict(bad), "evaluation": evaluate(bad, expected)}}


def save_csv(path: Path, rows: list[dict]) -> None:
    with path.open("w", encoding="utf-8", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=list(rows[0]), lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)


def experiment(output: Path) -> None:
    probe = examples()
    official, noise, questions, gold = build()
    current = {q.slots[0][0]: LATEST for q in questions if q.intent == "navigate"}
    buckets = defaultdict(list)
    rows = []
    for profile, passages in (("official_only", official), ("mixed", official + noise)):
        index = Index(passages)
        for question in questions:
            first = index.search(question.text, TOP_K)
            wide = index.search(question.text, WIDE_K)
            expected = tuple(gold.get(slot) for slot in question.slots)
            for policy in POLICIES:
                out = run(policy, question, index, current, first, wide)
                row = {
                    "profile": profile, "question": question.id, "intent": question.intent, "policy": policy,
                    "expected": "|".join(g or "-" for g in expected), "output": out.output,
                    "citations": "|".join(out.citations), "status": evaluate(out, expected),
                    "calls": out.calls, "returned": out.returned,
                }
                buckets[profile, "all", policy].append(row)
                buckets[profile, question.intent, policy].append(row)
                rows.append(row)
    summary = []
    for (profile, intent, policy), group in sorted(buckets.items()):
        counts = Counter(row["status"] for row in group)
        summary.append({
            "profile": profile, "intent": intent, "policy": policy, "n": len(group),
            **{name: counts[name] for name in ("covered", "missed", "unscored_links", "grounded",
                                                "wrong_answer", "missed_answer", "correct_abstain")},
            "mean_searches": round(mean(row["calls"] for row in group), 4),
            "mean_returned": round(mean(row["returned"] for row in group), 4),
        })
    output.mkdir(parents=True, exist_ok=True)
    save_csv(output / "summary.csv", summary)
    save_csv(output / "by_query.csv", rows)
    corpus = {"official": [asdict(p) for p in official], "noise": [asdict(p) for p in noise],
              "questions": [asdict(q) for q in questions]}
    raw = json.dumps(corpus, ensure_ascii=False, sort_keys=True).encode("utf-8")
    with (output / "corpus.json.gz").open("wb") as file:
        with gzip.GzipFile(filename="", mode="wb", fileobj=file, mtime=0) as compressed:
            compressed.write(raw)
    files = ("summary.csv", "by_query.csv", "corpus.json.gz")
    metadata = {"seed": SEED, "top_k": TOP_K, "wide_k": WIDE_K, "questions": len(questions),
                "question_types": Counter(q.intent for q in questions), "official_passages": len(official),
                "noise_passages": len(noise), "strategy_results": len(rows), "examples": probe,
                "sha256": {name: hashlib.sha256((output / name).read_bytes()).hexdigest()
                           for name in files}}
    (output / "run.json").write_text(json.dumps(metadata, ensure_ascii=False, indent=2) + "\n",
                                     encoding="utf-8", newline="\n")
    print(f"{len(questions)} paired questions; {len(rows)} strategy outputs.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", type=Path, default=Path("results"))
    experiment(parser.parse_args().out)
