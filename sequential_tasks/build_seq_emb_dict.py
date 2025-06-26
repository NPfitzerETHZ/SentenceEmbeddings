"""
Extended-sequence builder
~~~~~~~~~~~~~~~~~~~~~~~~~
Create a dictionary identical to `Seq` but with an **Embeddings** key whose
entries are chosen *once per unique state* and then reused wherever that state
re-appears.

Author : you
Python : ≥3.9
"""

from __future__ import annotations

import random
from copy import deepcopy
from typing import Any, Dict, Hashable, List


def build_seq_emb(
    seq: Dict[str, List[Any]],
    dataset: Dict[Hashable, List[Any]],
    rng_seed: int | None = None,
) -> Dict[str, List[Any]]:
    """
    Parameters
    ----------
    seq : dict
        Must contain keys ``"Events"`` (list[tuple]) and ``"States"`` (list[label]).
    dataset : dict
        Maps each state label -> list[embedding]  (≈1 000 vectors per label).
    rng_seed : int | None, optional
        Fix a seed for deterministic sampling.

    Returns
    -------
    dict
        Deep-copied ``seq`` plus an ``"Embeddings"`` list that is
        position-aligned with ``seq["States"]``.
    """
    if rng_seed is not None:
        random.seed(rng_seed)

    # ------------------------------------------------------------------
    # 1. One random embedding per unique state label
    # ------------------------------------------------------------------
    state_to_embedding: Dict[Hashable, Any] = {}
    for state in seq["States"]:
        if state not in state_to_embedding:
            state_to_embedding[state] = random.choice(dataset[state])

    # ------------------------------------------------------------------
    # 2. Broadcast those embeddings across the sequence
    # ------------------------------------------------------------------
    embeddings: List[Any] = [state_to_embedding[s] for s in seq["States"]]

    # ------------------------------------------------------------------
    # 3. Assemble the extended dictionary
    # ------------------------------------------------------------------
    seq_emb = deepcopy(seq)
    seq_emb["Embeddings"] = embeddings
    return seq_emb


# ----------------------------------------------------------------------
# Example usage (remove or adapt as needed)
# ----------------------------------------------------------------------
if __name__ == "__main__":
    # Minimal toy data
    Seq = {
        "Events": [(0, 0, 0), (0, 1, 0), (1, 0, 0), (0, 0, 1)],
        "States": ["E", "E", "N", "N", "F"],
    }

    # Pretend each vector is a tiny embedding
    Dataset = {
        "E": [[0.1, 0.2, 0.3], [0.4, 0.5, 0.6]],
        "N": [[1.0, 1.1, 1.2], [1.3, 1.4, 1.5]],
        "F": [[2.0, 2.1, 2.2], [2.3, 2.4, 2.5]],
    }

    extended = build_seq_emb(Seq, Dataset, rng_seed=42)
    print(extended)
