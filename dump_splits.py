"""Write the contract titles of each split to dataset/splits/.

The benchmark team needs these to exclude training contracts from their
suites (contamination check). val.txt reproduces prepare_data.py's split
exactly: same seed, same shuffle, constants imported so they cannot drift.
"""

import json
import random
from pathlib import Path

from prepare_data import SEED, SRC, VAL_CONTRACTS

OUT = Path("dataset/splits")

contracts = json.load(SRC.open())["data"]
rng = random.Random(SEED)
rng.shuffle(contracts)
splits = {
    "val": [c["title"] for c in contracts[:VAL_CONTRACTS]],
    "train": [c["title"] for c in contracts[VAL_CONTRACTS:]],
    "test": [c["title"] for c in json.load(open("dataset/cuad-split/test.json"))["data"]],
}

OUT.mkdir(parents=True, exist_ok=True)
for name, titles in splits.items():
    (OUT / f"{name}.txt").write_text("\n".join(titles) + "\n")
    print(f"{name}: {len(titles)} contracts")
