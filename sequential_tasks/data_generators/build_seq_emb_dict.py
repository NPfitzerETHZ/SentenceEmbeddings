#!/usr/bin/env python
"""
seq_emb_builder_fixed.py
~~~~~~~~~~~~~~~~~~~~~~~~
• Reads ONE sequence file: a list of {"events": [...], "states": [...]} dicts  
• Reads ONE dataset file:  {"E": [{"response": "...", "embedding": [...]}, …], …}  
• Writes ONE output file:  the same list, each dict now has "responses" and "embeddings"

All file paths are set in the CONFIG section—no command-line flags.
"""

from __future__ import annotations
import json, random, sys
from copy import deepcopy
from pathlib import Path
from typing import Any, Dict, Hashable, List, Tuple
from collections import Counter

# ───────────────────────── CONFIG ───────────────────────── #
SEQ_JSON      = Path("sequential_tasks/data/random_walk_flag_home_results.json")       # ← edit me
DATASET_JSON  = Path("sequential_tasks/data/merged.json")   # ← edit me
OUTPUT_JSON   = Path("sequential_tasks/data/dataset_no_summary_flag_home.json")  # ← edit me or set to None
RNG_SEED      = 42                          # set None for non-deterministic
# ─────────────────────────────────────────────────────────── #

def _sample_one(state: Hashable,
                pool: List[Dict[str, Any]]) -> Tuple[str, Any]:
    """Pick one `{response, embedding}` for *state*."""
    if not pool:
        raise ValueError(f"Dataset entry for state {state!r} is empty")
    rec = random.choice(pool)
    if "response" not in rec or "embedding" not in rec:
        raise ValueError(f"Object for {state!r} lacks 'response' or 'embedding'")
    return rec["response"], rec["embedding"]

def extend_sequences(
    seq_list: List[Dict[str, List[Any]]],
    dataset: Dict[Hashable, List[Dict[str, Any]]],
) -> List[Dict[str, List[Any]]]:
    """Return new list with `responses` and `embeddings` added to every item."""
    # one random pair per unique state
    unique_states = [
        {state                       # or use [...] if you prefer a list
        for state, n in Counter(seq["states"]).items()}                  # keep states that appear exactly once
        for seq in seq_list
    ]
    state_to_pair = [{
        state: _sample_one(state, dataset[state]) for state, n in Counter(seq["states"]).items()}                  # keep states that appear exactly once
        for seq in seq_list 
    ]

    extended: List[Dict[str, Any]] = []
    for i, seq in enumerate(seq_list):
        responses  = [state_to_pair[i][s][0] for s in seq["states"]]
        embeddings = [state_to_pair[i][s][1] for s in seq["states"]]
        new_seq = deepcopy(seq)
        new_seq["responses"]  = responses
        new_seq["embeddings"] = embeddings
        extended.append(new_seq)
    return extended

def main() -> None:
    if RNG_SEED is not None:
        random.seed(RNG_SEED)

    with SEQ_JSON.open(encoding="utf-8") as f:
        seq_list = json.load(f)

    with DATASET_JSON.open(encoding="utf-8") as f:
        dataset = json.load(f)

    extended = extend_sequences(seq_list, dataset)

    if OUTPUT_JSON is None:
        json.dump(extended, sys.stdout, indent=2, sort_keys=True)
        sys.stdout.write("\n")
    else:
        OUTPUT_JSON.parent.mkdir(parents=True, exist_ok=True)
        with OUTPUT_JSON.open("w", encoding="utf-8") as f:
            json.dump(extended, f, indent=2, sort_keys=True)
        print(f"[INFO] Extended sequences written to {OUTPUT_JSON.resolve()}")

if __name__ == "__main__":
    main()
