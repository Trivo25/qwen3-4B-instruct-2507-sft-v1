# cuad-qwen3 — model card and interface contract

Finetune of Qwen/Qwen3-4B-Instruct-2507 for CUAD legal-clause extraction.
Served in ollama as `cuad-qwen3` (merged LoRA, GGUF q8_0, see Modelfile).
This file is the contract for anyone calling the model. Date: 2026-09-01.

## Interface (v2, the currently served model)

System prompt: baked into the Modelfile, byte-identical to training. Do not
send your own system message.

User message, exact layout (question FIRST — the model was trained on this
one layout and misses when the excerpt comes first):

```
{original CUAD category question}

Contract excerpt:
{excerpt}
```

- Excerpt limit: ~1,800 words (~2,500 tokens). NEVER send a full contract:
  inputs far outside the training distribution make the model fall back to
  memorized training clauses instead of `None` (observed: 0% recall on
  full-contract inputs). Window the contract and aggregate — see
  `windows()` in prepare_data.py (1,800 words, 50% overlap).
- Output grammar: verbatim clause passages, one per line — or exactly
  `None`. No citations, no prose (that is v3, below).
- Decoding: temperature 0, stop `<|im_end|>`, num_ctx >= 4096
  (the Modelfile sets all three).

## Training data and contamination

Trained on the official CUAD train split ONLY (both v2 and v3):

- `dataset/splits/train.txt` — 388 contracts, trained on
- `dataset/splits/val.txt` — 20 contracts, validation (same source file;
  treat as seen)
- `dataset/splits/test.txt` — 102 contracts (official CUAD test split),
  UNSEEN — the only contracts valid for benchmarking

Any benchmark case built from a train.txt or val.txt contract measures
memorization, not skill, and must be excluded. The memorized "Oceanic"
liquidated-damages answer observed in full-contract probes likely traces to
`STARTECGLOBALCOMMUNICATIONSCORP_11_16_1998-...` (train split, text
mentions Oceanic).

## Training summary (v2)

QLoRA (r=16, alpha=32, all linear layers, 33M trainable params) on 20,010
window examples (1:1 positive/negative), 2 epochs, unsloth + TRL,
sequence length 5,120, effective batch 8, lr 2e-4 cosine. Val loss
0.060 -> 0.051 -> 0.049 (falling throughout; no overfitting signal at the
loss level — the memorization above is verbatim-recall of distinctive
training text, which loss curves do not catch).

## v3 (in training): numbered passages + citations

Interface the next adapter is trained for, so harnesses can prepare:

User message (75% of training uses question-first; 25% passages-first, so
both layouts work):

```
{question}

Passages:
[1] {~200-word passage}
[2] {...}
...up to [8]
```

Output: one line per found clause, `[n] "verbatim quote"` — with citation
chains `[2][3]` when a clause crosses passages — or exactly:
`No passage answers this question.`
