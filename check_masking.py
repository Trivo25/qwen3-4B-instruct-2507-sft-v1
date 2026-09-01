"""Verify unsloth's loss masking before spending GPU hours.

Builds the real trainer via train_unsloth.build_trainer() and checks the
labels it produced: for every sampled example, the trained tokens must be
exactly the assistant answer plus its closing <|im_end|>, nothing else.
Equivalent of axolotl's `preprocess --debug` check for run #1.
"""

import random

from prepare_data_v3 import NO_ANSWER
from train_unsloth import build_trainer

N_SAMPLES = 30

trainer, tokenizer = build_trainer()
ds = trainer.train_dataset
rng = random.Random(0)
indices = rng.sample(range(len(ds)), N_SAMPLES)

im_end = "<|im_end|>"
failures = 0
shown = {"positive": False, "negative": False}

for i in indices:
    row = ds[i]
    input_ids, labels = row["input_ids"], row["labels"]
    trained = [t for t, l in zip(input_ids, labels) if l != -100]
    trained_text = tokenizer.decode(trained)

    # ground truth: everything after the last assistant header in the
    # rendered text, which is answer + <|im_end|> (+ trailing newline)
    full_text = tokenizer.decode(input_ids)

    # template contamination guard: comparing trained vs rendered text can't
    # see a poisoned template, so check the rendering itself for think tags
    if "<think>" in full_text:
        failures += 1
        print(f"FAIL idx={i}: <think> found in rendered text")
        continue

    expected = full_text.rsplit("<|im_start|>assistant\n", 1)[-1]
    ok = trained_text.strip() == expected.strip() and trained_text.rstrip().endswith(im_end)

    if not ok:
        failures += 1
        print(f"FAIL idx={i}")
        print(f"  trained:  {trained_text!r}")
        print(f"  expected: {expected!r}")

    # show one worked example of each kind for eyeballing
    kind = "negative" if expected.strip().startswith(NO_ANSWER) else "positive"
    if ok and not shown[kind]:
        shown[kind] = True
        print(f"--- {kind} example idx={i} ---")
        print(f"  tokens total={len(input_ids)} trained={len(trained)}")
        print(f"  trained text: {trained_text[:300]!r}")

print(f"\nchecked {N_SAMPLES} examples: {N_SAMPLES - failures} ok, {failures} failed")
if failures:
    raise SystemExit(1)
print("masking OK — safe to train")
