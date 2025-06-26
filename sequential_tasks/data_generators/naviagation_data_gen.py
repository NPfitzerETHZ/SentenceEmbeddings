#!/usr/bin/env python3
"""
generate_prompts.py
-------------------
Collect **N** return‑to‑base prompts and save them as a JSON array.

Output format
-------------
[
  {"response": "prompt 1"},
  {"response": "prompt 2"},
  ...
]

Usage examples
--------------
# 50 prompts to the default file
python generate_prompts.py -n 50

# 250 prompts into a custom file
python generate_prompts.py -n 250 -o my_prompts.json
"""
from __future__ import annotations

import argparse
import json
import random
from pathlib import Path
from typing import List

_LEADERS: List[str] = [
    "All units",
    "Expedition team",
    "Recon squad",
    "Exploration swarm",
    "Systems collective",
    "Task force",
    "Survey cohort",
    "Recovery detachment",
]

_VERBS: List[str] = [
    "return",
    "proceed",
    "navigate",
    "head",
    "travel",
    "vector",
    "route",
    "advance",
]

_ADVERBS: List[str] = [
    "promptly",
    "efficiently",
    "directly",
    "swiftly",
    "smoothly",
    "without delay",
    "expeditiously",
    "securely",
]

_BASES: List[str] = [
    "base",
    "home station",
    "operations hub",
    "charging depot",
    "command post",
    "control nexus",
    "staging platform",
    "operations center",
]

_ROUTE_PHRASES: List[str] = [
    "Select the safest corridor your sensors identify.",
    "Choose the most power‑efficient route you can compute.",
    "Bypass any dynamic obstacles detected along the path.",
    "Avoid hazardous terrain and recalibrate as needed.",
    "Use on‑board mapping to optimize distance and energy.",
]

_ARRIVAL_PHRASES: List[str] = [
    "Upon arrival, initiate data offload and recharge protocols.",
    "On reaching base, commence status checks and diagnostics.",
    "When you arrive, sync logs with the master server.",
    "After docking, begin maintenance routines.",
]

_GUIDELINES: List[str] = [
    "Maintain formation integrity and 2‑meter spacing.",
    "Share real‑time telemetry with the fleet network.",
    "Keep lidar and vision stacks active for situational awareness.",
    "Conserve battery reserves where possible.",
    "Monitor environmental variables for unexpected changes.",
    "Communicate any anomalies immediately.",
]


# --------------------------- public interface ---------------------------

def return_to_base_prompt() -> str:
    """Generate a rich natural‑language command instructing robots to return home.

    The prompt is plain text—no markdown, no special formatting—suitable
    for feeding straight into Gemini or any LLM.
    """
    lines: List[str] = []

    # Core command
    lines.append(
        f"{random.choice(_LEADERS)}, {random.choice(_VERBS)} {random.choice(_ADVERBS)} "
        f"to the {random.choice(_BASES)}."
    )

    # # Route & arrival context
    # lines.append(random.choice(_ROUTE_PHRASES))
    # lines.append(random.choice(_ARRIVAL_PHRASES))

    # # Add two random guidelines for richness
    # lines.extend(random.sample(_GUIDELINES, k=2))

    return " ".join(lines)


def collect_prompts(n: int) -> list[dict[str, str]]:
    """Generate *n* distinct prompts wrapped in {"response": ...} objects."""
    return [{"response": return_to_base_prompt()} for _ in range(n)]


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Generate N return-to-base prompts and dump to JSON."
    )
    parser.add_argument(
        "-n",
        "--num",
        type=int,
        default=1000,
        help="number of prompts to generate (default: 100)",
    )
    parser.add_argument(
        "-o",
        "--output",
        type=Path,
        default=Path("sequential_tasks/sentences/navigation_sentences.json"),
        help='output JSON file (default: "sequential_tasks/sentences/navigation_sentences.json")',
    )

    args = parser.parse_args()

    prompts = collect_prompts(args.num)

    # Write the list as a compact, UTF‑8 encoded JSON array
    args.output.write_text(
        json.dumps(prompts, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(f"Wrote {args.num} prompts → {args.output.resolve()}")


if __name__ == "__main__":
    main()

