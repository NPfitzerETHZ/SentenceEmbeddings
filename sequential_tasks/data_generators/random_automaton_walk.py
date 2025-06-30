"""merged_random_walks.py
---------------------------------
Run identical random‑walk experiments for multiple DFAs.

⚙ **Centralised experiment parameters**
   Edit *once* at the top (``MIN_STEPS``, ``MAX_STEPS``, ``REPS``, ``RNG_SEED``) and they apply to *all* DFAs.

Other features stay the same:
    • Each DFA described once in a lightweight `DFAConfig`.
    • Every record still has `events`, `states`, `success`.
"""

from __future__ import annotations

import json
import random
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Dict, List, Sequence, Tuple

# --------------------------------------------------------------------------- #
#  Global experiment parameters – tweak them here
# --------------------------------------------------------------------------- #

MIN_STEPS: int = 2   # inclusive lower bound for random sequence length L
MAX_STEPS: int = 8  # inclusive upper bound for L
REPS: int = 500        # generate REPS successes + REPS failures per DFA
RNG_SEED: int | None = 42  # None ➟ do not reset RNG

# --------------------------------------------------------------------------- #
#  Generic DFA machinery
# --------------------------------------------------------------------------- #

EVENTS: Sequence[str] = ("a", "b", "c")              # index 0,1,2 ↔ a,b,c
Vector = List[int]                                      # e.g. [0, 1, 0]
State = str
Trace = List[Tuple[Vector, State]]                     # [(vec, next_state), …]


class Automaton:
    def __init__(
        self,
        transition: Dict[State, Callable[[Vector], State]],
        initial: State,
        finals: set[State],
    ) -> None:
        self._transition = transition
        self._initial = initial
        self._finals = finals

    # ----------------------- core DFA API -----------------------

    def step(self, state: State, vec: Vector) -> State:
        return self._transition[state](vec)

    def run(self, vectors: Sequence[Vector]) -> Tuple[Trace | List[State], bool]:
        state: State = self._initial
        trace: List[State | Tuple[Vector, State]] = [state]
        for v in vectors:
            state = self.step(state, v)
            trace.append((v, state))
        return trace, (state in self._finals)


# --------------------------------------------------------------------------- #
#  Experiment helpers
# --------------------------------------------------------------------------- #


def rand_vec() -> Vector:
    return [random.randint(0, 1) for _ in EVENTS]


def random_walk(
    automaton: Automaton,
    steps: int,
    accept: bool,
    *,
    max_attempts: int = 1_000,
) -> Trace:
    for _ in range(max_attempts):
        vecs = [rand_vec() for _ in range(steps)]
        trace, ok = automaton.run(vecs)
        if ok == accept:
            return trace
    raise RuntimeError("Could not satisfy requested outcome in random_walk")


# JSON conversion -----------------------------------------------------------


def record_trace(trace: List[State | Tuple[Vector, State]], accepted: bool) -> dict:
    events = [vec for (vec, _) in trace[1:]]
    states = [trace[0]] + [st for (_, st) in trace[1:]]
    return {"events": events, "states": states, "success": accepted}


# --------------------------------------------------------------------------- #
#  Mini‑DSL: describe each DFA in one @dataclass (no experiment params here!)
# --------------------------------------------------------------------------- #


@dataclass(frozen=True)
class DFAConfig:
    name: str
    initial: State
    finals: set[State]
    transition: Dict[State, Callable[[Vector], State]]
    outfile: Path


# --------------------------------------------------------------------------- #
#  Concrete DFA definitions
# --------------------------------------------------------------------------- #


def flag_home_config() -> DFAConfig:
    # States:  E → N → F
    def next_E(v):
        a, _, _ = v
        return "N" if a else "E"

    def next_N(v):
        _, b, c = v
        if c:
            return "F"
        return "N" if b else "E"

    def next_F(_):
        return "F"

    return DFAConfig(
        name="flag_home",
        initial="E",
        finals={"F"},
        transition={"E": next_E, "N": next_N, "F": next_F},
        outfile=Path("sequential_tasks/data/random_walk_flag_home_results.json"),
    )


def defend_config() -> DFAConfig:
    # States:  E → P1 → P2
    def next_E(v):
        a, _, _ = v
        return "P1" if a else "E"

    def next_P1(v):
        _, b, c = v
        if c:
            return "P2"
        return "P1" if b else "E"

    def next_P2(_):
        return "P2"

    return DFAConfig(
        name="defend",
        initial="E",
        finals={"P2"},
        transition={"E": next_E, "P1": next_P1, "P2": next_P2},
        outfile=Path("sequential_tasks/data/random_walk_defend_results.json"),
    )


# --------------------------------------------------------------------------- #
#  Master runner (uses GLOBAL parameters)
# --------------------------------------------------------------------------- #


def run_and_save(cfg: DFAConfig) -> None:
    if RNG_SEED is not None:
        random.seed(RNG_SEED)

    dfa = Automaton(cfg.transition, cfg.initial, cfg.finals)

    all_runs: List[dict] = []
    for accept in (True, False):
        for _ in range(REPS):
            L = random.randint(MIN_STEPS, MAX_STEPS)
            trace = random_walk(dfa, L, accept)
            all_runs.append(record_trace(trace, accept))

    cfg.outfile.parent.mkdir(parents=True, exist_ok=True)
    cfg.outfile.write_text(json.dumps(all_runs, indent=2))
    print(f"[{cfg.name}] Wrote {len(all_runs)} runs ➜ {cfg.outfile.resolve()}")


# --------------------------------------------------------------------------- #
#  Entry point
# --------------------------------------------------------------------------- #


if __name__ == "__main__":
    for cfg in (flag_home_config(), defend_config()):
        run_and_save(cfg)
