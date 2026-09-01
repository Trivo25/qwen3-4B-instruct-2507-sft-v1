"""Build SFT chat data for CUAD clause extraction.

Slices each contract of the official CUAD train split into overlapping
word windows and emits chat-format JSONL for axolotl. Positive examples
quote the clause spans inside the window verbatim; sampled negatives
answer "None". CUAD: https://arxiv.org/abs/2103.06268
"""

import json
import random
import re
from pathlib import Path

# window/stride in words; the overlap (900 words, ~1200 tokens) exceeds the
# longest span in the data (~640 tokens), so every span fits fully in at
# least one window
WINDOW = 1800
STRIDE = 900
NEG_RATIO = 1.0     # negatives sampled per positive, per contract
VAL_CONTRACTS = 20  # held out by contract, not by example, to avoid leakage
SEED = 0

SRC = Path("dataset/cuad-split/train_separate_questions.json")
OUT = Path("dataset/prepared")

SYSTEM = (
    "You are a legal contract analyst. You are given an excerpt from a "
    "commercial contract and one review question. Quote every part of the "
    "excerpt that answers the question, verbatim, one passage per line. "
    "If nothing in the excerpt answers the question, reply exactly: None"
)


def windows(context):
    """Yield (char_start, char_end) word windows over the contract text."""
    words = [m.span() for m in re.finditer(r"\S+", context)]
    for i in range(0, len(words), STRIDE):
        chunk = words[i : i + WINDOW]
        yield chunk[0][0], chunk[-1][1]
        if i + WINDOW >= len(words):
            break


def categories(paragraph):
    """Group span-level QAs back into one entry per clause category.

    train_separate_questions.json stores each annotated span as its own QA
    with ids like "TITLE__Parties_3"; regrouping avoids training the same
    prompt against several different "correct" answers.
    """
    cats = {}
    for qa in paragraph["qas"]:
        cat = re.sub(r"_\d+$", "", qa["id"].rsplit("__", 1)[-1])
        entry = cats.setdefault(cat, {"question": qa["question"], "spans": []})
        for a in qa["answers"]:
            entry["spans"].append((a["answer_start"], a["text"]))
    return cats


def example(question, excerpt, answer):
    return {
        "messages": [
            {"role": "system", "content": SYSTEM},
            {"role": "user", "content": f"{question}\n\nContract excerpt:\n{excerpt}"},
            {"role": "assistant", "content": answer},
        ]
    }


def build(contract, rng, stats):
    """All positives plus sampled negatives for one contract."""
    context = contract["paragraphs"][0]["context"]
    cats = categories(contract["paragraphs"][0])
    # offsets sanity check: the span text must match the context exactly
    for entry in cats.values():
        for start, text in entry["spans"]:
            if context[start : start + len(text)] != text:
                stats["bad_offsets"] += 1
    pos, neg_pool = [], []
    for w_start, w_end in windows(context):
        excerpt = context[w_start:w_end]
        for entry in cats.values():
            inside, partial = [], False
            for start, text in entry["spans"]:
                end = start + len(text)
                if w_start <= start and end <= w_end:
                    inside.append((start, text))
                elif start < w_end and end > w_start:
                    partial = True  # span cut by the window edge
            if inside:
                inside.sort()
                pos.append(example(entry["question"], excerpt, "\n".join(t for _, t in inside)))
            elif not partial:
                # true negative: no trace of this category in the window;
                # windows holding only a cut span are dropped as ambiguous
                neg_pool.append((entry["question"], excerpt))
    n_neg = min(len(neg_pool), round(len(pos) * NEG_RATIO))
    neg = [example(q, e, "None") for q, e in rng.sample(neg_pool, n_neg)]
    stats["pos"] += len(pos)
    stats["neg"] += len(neg)
    return pos + neg


def main():
    rng = random.Random(SEED)
    contracts = json.load(SRC.open())["data"]
    rng.shuffle(contracts)
    splits = {"val": contracts[:VAL_CONTRACTS], "train": contracts[VAL_CONTRACTS:]}
    OUT.mkdir(parents=True, exist_ok=True)
    for name, subset in splits.items():
        stats = {"pos": 0, "neg": 0, "bad_offsets": 0}
        rows = []
        for contract in subset:
            rows.extend(build(contract, rng, stats))
        rng.shuffle(rows)
        with (OUT / f"{name}.jsonl").open("w") as f:
            for row in rows:
                f.write(json.dumps(row, ensure_ascii=False) + "\n")
        # rough token estimate (~4/3 tokens per word) to size sequence_len
        toks = sorted(
            len(json.dumps(r["messages"]).split()) * 4 // 3 for r in rows
        )
        print(
            f"{name}: contracts={len(subset)} examples={len(rows)} "
            f"pos={stats['pos']} neg={stats['neg']} bad_offsets={stats['bad_offsets']} "
            f"est_tokens median={toks[len(toks) // 2]} p99={toks[int(len(toks) * 0.99)]} max={toks[-1]}"
        )


if __name__ == "__main__":
    main()
