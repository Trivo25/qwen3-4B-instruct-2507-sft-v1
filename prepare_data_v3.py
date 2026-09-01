"""Build v3 SFT data: numbered passages, [n]-cited verbatim quotes.

Changes vs prepare_data.py (v1, quote-or-None):
- contracts split into ~200-word numbered passages; each prompt holds 8
  consecutive passages (stride 4 = 50% overlap), numbered [1..8] locally
- targets cite passages: [2] "quote" ([2][3] chains when a span crosses)
- absence answer is a sentence: No passage answers this question.
- 25% of prompts put the passages before the question (layout robustness)
See MODEL_CARD.md "v3" for the resulting interface.
"""

import json
import random
import re
from pathlib import Path

# same split constants and category grouping as v1 — val split stays identical
from prepare_data import SEED, SRC, VAL_CONTRACTS, categories

PASSAGE_WORDS = 200
PER_PROMPT = 8   # passages per prompt (~1600 words)
STRIDE = 4
NEG_RATIO = 1.0
OUT = Path("dataset/prepared-v3")

SYSTEM = (
    "You are a legal contract analyst. You are given numbered passages from "
    "a commercial contract and one review question. Quote every part of the "
    "passages that answers the question, verbatim, one quote per line in "
    'the form [n] "quote", where [n] cites the passage the quote comes '
    "from. If no passage answers the question, reply exactly: No passage "
    "answers this question."
)
NO_ANSWER = "No passage answers this question."


def passages(context):
    """~PASSAGE_WORDS-word passages as (char_start, char_end)."""
    words = [m.span() for m in re.finditer(r"\S+", context)]
    out = []
    for i in range(0, len(words), PASSAGE_WORDS):
        chunk = words[i : i + PASSAGE_WORDS]
        out.append((chunk[0][0], chunk[-1][1]))
    return out


def example(question, numbered, answer, rng):
    # 25% passages-first so the model is not locked to one layout
    if rng.random() < 0.25:
        user = f"Passages:\n{numbered}\n\nQuestion: {question}"
    else:
        user = f"{question}\n\nPassages:\n{numbered}"
    return {"messages": [
        {"role": "system", "content": SYSTEM},
        {"role": "user", "content": user},
        {"role": "assistant", "content": answer},
    ]}


def build(contract, rng, stats):
    context = contract["paragraphs"][0]["context"]
    cats = categories(contract["paragraphs"][0])
    p = passages(context)
    # map each span to the range of passage indices it overlaps
    for entry in cats.values():
        entry["ranges"] = []
        for start, text in entry["spans"]:
            end = start + len(text)
            idx = [i for i, (ps, pe) in enumerate(p) if ps < end and pe > start]
            if idx:
                entry["ranges"].append((idx[0], idx[-1], start, text))

    pos, neg_pool = [], []
    for w0 in range(0, len(p), STRIDE):
        w1 = min(w0 + PER_PROMPT, len(p))
        numbered = "\n".join(
            f"[{i - w0 + 1}] {context[s:e]}" for i, (s, e) in enumerate(p[w0:w1], start=w0)
        )
        for entry in cats.values():
            inside, partial = [], False
            for first, last, start, text in entry["ranges"]:
                if w0 <= first and last < w1:
                    inside.append((first, last, start, text))
                elif first < w1 and last >= w0:
                    partial = True  # span cut by the window edge: ambiguous
            if inside:
                inside.sort(key=lambda r: (r[0], r[2]))
                lines = [
                    "".join(f"[{i - w0 + 1}]" for i in range(first, last + 1))
                    + f' "{text}"'
                    for first, last, _, text in inside
                ]
                pos.append(example(entry["question"], numbered, "\n".join(lines), rng))
            elif not partial:
                neg_pool.append((entry["question"], numbered))
        if w1 == len(p):
            break

    n_neg = min(len(neg_pool), round(len(pos) * NEG_RATIO))
    neg = [example(q, n, NO_ANSWER, rng) for q, n in rng.sample(neg_pool, n_neg)]
    stats["pos"] += len(pos)
    stats["neg"] += len(neg)
    return pos + neg


def main():
    rng = random.Random(SEED)
    contracts = json.load(SRC.open())["data"]
    rng.shuffle(contracts)  # same seed+order as v1: identical val contracts
    splits = {"val": contracts[:VAL_CONTRACTS], "train": contracts[VAL_CONTRACTS:]}
    OUT.mkdir(parents=True, exist_ok=True)
    for name, subset in splits.items():
        stats = {"pos": 0, "neg": 0}
        rows = []
        for contract in subset:
            rows.extend(build(contract, rng, stats))
        rng.shuffle(rows)
        with (OUT / f"{name}.jsonl").open("w") as f:
            for row in rows:
                f.write(json.dumps(row, ensure_ascii=False) + "\n")
        toks = sorted(
            len(json.dumps(r["messages"]).split()) * 4 // 3 for r in rows
        )
        print(
            f"{name}: examples={len(rows)} pos={stats['pos']} neg={stats['neg']} "
            f"est_tokens median={toks[len(toks) // 2]} p99={toks[int(len(toks) * 0.99)]} max={toks[-1]}"
        )


if __name__ == "__main__":
    main()
