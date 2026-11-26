"""Dataset-level demonstration selection. The selector receives one support pool."""
import argparse
import hashlib
import json
import re
from collections import Counter
from pathlib import Path

import numpy as np
from scipy.sparse import csr_matrix, hstack
from sklearn.feature_extraction.text import TfidfVectorizer


def infer_kind(rows, kind):
    if kind != "auto":
        return kind
    if all(isinstance(r["output"], bool) for r in rows):
        return "classification"
    if all("class_id" in r for r in rows):
        return "classification"
    if any(r.get("options") for r in rows):
        options = [r.get("options") or [] for r in rows]
        if all(isinstance(a, (list, tuple)) and all(isinstance(v, str) for v in a)
               for a in options):
            vocab = set(options[0])
            slot = r"(?:option\s*|choice\s*)?\(?[a-z0-9]\)?[.:]?|\d+"
            if (len(vocab) >= 2 and all(set(a) == vocab for a in options)
                    and not any(re.fullmatch(slot, v.strip(), re.I) for v in vocab)
                    and all(isinstance(r["output"], str) and r["output"] in vocab for r in rows)):
                return "classification"
        return "local_choice"
    return "open"


def category(row):
    value = row.get("class_id", row["output"])
    if value is None or not isinstance(value, (str, int, float, bool)):
        raise ValueError("Supply one scalar global class per labeled support row.")
    return json.dumps(value, ensure_ascii=False, allow_nan=False)


def features(texts, option_counts):
    # Empty or constant feature families contribute zero to the distance.
    text = TfidfVectorizer(analyzer="char", ngram_range=(1, 4),
                           max_features=4096, dtype=np.float64).fit_transform(texts)
    lengths = np.array([len(t) for t in texts])
    edges = np.unique(np.quantile(lengths, [.25, .5, .75]))
    lens = np.searchsorted(edges, lengths, side="left")
    _, counts = np.unique(option_counts, return_inverse=True)
    def one_hot(indices):
        return csr_matrix((np.ones(len(indices)), (np.arange(len(indices)), indices)))
    return hstack([text, one_hot(lens), one_hot(counts)], format="csr") / np.sqrt(3)


def quotas(counts, k, equal=False):
    totals = {y: k * (1 if equal else n) for y, n in counts.items()}
    denom = len(counts) if equal else sum(counts.values())
    q = dict.fromkeys(counts, 0)
    for _ in range(k):
        available = (y for y in q if q[y] < counts[y])
        y = min(available, key=lambda y: (q[y] * denom - totals[y], y))
        q[y] += 1
    return q


class Selector:
    def __init__(self, pool, kind="auto"):
        if kind not in {"auto", "classification", "local_choice", "open"}:
            raise ValueError("Unsupported schema kind")
        if not pool or any(not isinstance(r.get("id"), str) or not r["id"] or
                           not isinstance(r.get("input"), str) or not r["input"].strip()
                           or r.get("output") is None for r in pool):
            raise ValueError("Each support example needs a string ID, text input and output.")
        self.rows = sorted(pool, key=lambda r: r["id"])
        self.ids = [r["id"] for r in self.rows]
        if len(set(self.ids)) != len(self.ids):
            raise ValueError("Example IDs must be unique.")
        self.kind = infer_kind(self.rows, kind)
        self.fixed = self.kind == "classification"
        if self.fixed and any("class_id" in r for r in self.rows) and not all(
                "class_id" in r for r in self.rows):
            raise ValueError("Supply class_id for every row or omit it everywhere.")
        self.labels = [category(r) if self.fixed else "*" for r in self.rows]
        self.counts = Counter(self.labels)
        # Option texts may carry most of a question's information. No gold marker is rendered.
        texts, n_opts = [], []
        for r in self.rows:
            opts = r.get("options") or []
            if not isinstance(opts, (list, tuple)) or any(not isinstance(o, str) for o in opts):
                raise ValueError("options must be an array of strings")
            texts.append("\n".join([r["input"], *opts]))
            n_opts.append(len(opts))
        self.matrix = features(texts, n_opts)
        by_label = {}
        for i, y in enumerate(self.labels):
            by_label.setdefault(y, []).append(i)
        self.centers = {y: np.asarray(self.matrix[ix].mean(axis=0)).ravel()
                        for y, ix in by_label.items()}

    def select(self, k, seed=0, candidates=32):
        n = len(self.ids)
        if not 1 <= k <= n or candidates < 1:
            raise ValueError("Require 1 <= k <= pool size and candidates >= 1.")

        def rank(stream):
            def key(i):
                payload = json.dumps([seed, stream, self.ids[i]], ensure_ascii=False).encode()
                return hashlib.sha256(payload).digest()
            return sorted(range(n), key=key)

        def take(order, q):
            slots = q.copy()
            picked = []
            for i in order:
                y = self.labels[i]
                if slots[y] > 0:
                    picked.append(i)
                    slots[y] -= 1
                if len(picked) == k:
                    break
            return picked

        def distance(picked):
            groups = {}
            for i in picked:
                groups.setdefault(self.labels[i], []).append(i)
            value = 0.
            for y, ix in groups.items():
                mean = np.asarray(self.matrix[ix].mean(axis=0)).ravel()
                diff = mean - self.centers[y]
                value += self.counts[y] / n * float(diff @ diff)
            return value

        order = rank("draw")
        display = rank("display")
        groups = {"A": order[:k]}
        q = quotas(self.counts, k)
        if self.fixed:
            groups["B"] = take(order, quotas(self.counts, k, equal=True))
            groups["C"] = take(order, q)
        first = groups["C"] if self.fixed else groups["A"]
        alternatives = [first]
        if k < n:
            alternatives.extend(take(rank(f"cand-{j}"), q) for j in range(1, candidates))
        groups["D"] = min(alternatives, key=distance)
        result = {}
        for arm, picked in groups.items():
            kept = set(picked)
            hist = Counter(self.labels[i] for i in picked)
            result[arm] = {
                "ids": [self.ids[i] for i in display if i in kept],
                "label_counts": dict(hist) if self.fixed else None,
                "label_tv": (sum(abs(hist[y]/k - self.counts[y]/n)
                                 for y in self.counts) / 2 if self.fixed else None),
                "feature_distance": distance(picked),
            }
        return {"kind": self.kind, "n": n, "k": k, "seed": seed,
                "pool_counts": dict(self.counts) if self.fixed else None,
                "quota": q if self.fixed else None, "arms": result}


def main():
    p = argparse.ArgumentParser()
    p.add_argument("pool", type=Path, help="JSONL of support examples")
    p.add_argument("--shots", type=int, required=True)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--kind", default="auto", choices=["auto","classification","local_choice","open"])
    a = p.parse_args()
    rows = [json.loads(s) for s in a.pool.read_text().splitlines() if s.strip()]
    print(json.dumps(Selector(rows, a.kind).select(a.shots, a.seed),
                     ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
