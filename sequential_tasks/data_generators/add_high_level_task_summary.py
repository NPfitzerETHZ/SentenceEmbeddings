#!/usr/bin/env python
"""
summarise_seq_with_gemini_stream.py
-----------------------------------
• Streams one sequence-object at a time from EXTENDED_JSON
• Uses Gemini Flash 2.0 to generate a ONE-sentence summary of its responses
• Streams the augmented object straight into OUTPUT_JSON (an array)
  so memory usage stays almost constant.

Set GOOGLE_API_KEY in your environment.

© 2025 – minimal, no-CLI edition (now with progress reporting).
"""

from __future__ import annotations

import json
import os
import sys
import time
from pathlib import Path
from typing import List
from tqdm import tqdm

import ijson
import textwrap
from google import genai
from sentence_transformers import SentenceTransformer

# Allow project-root imports (for api_keys)
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from api_keys import GEMINI_API_KEY  # noqa: E402

# ========================== Configuration ==========================

# Gemini API Key
genai_client = genai.Client(api_key=GEMINI_API_KEY)  # 🔐 Replace with your actual key

# ─────────────── CONFIG ─────────────── #
EXTENDED_JSON = Path("sequential_tasks/data/dataset_no_summary_flag_home.json")  # <- input (large)
OUTPUT_JSON = Path("sequential_tasks/data/dataset_full.json")  # <- output
MODEL_NAME = "gemini-2.0-flash"
TEMP = 0.2  # lower = steadier summaries
PRINT_EVERY = 100  # heartbeat cadence (in items)
# ─────────────────────────────────────── #

PROMPT_TMPL = textwrap.dedent(
    """\
    You are leading a team of robots in  a capture the flag mission. Your team must find the flag and then return to base.    
    Write ONE sentence summarising the sequence of tasks. Ignore repetitions. Do **not** add anything else. The task sequence might mention an area of interest for the flag location.
    TASK SEQUENCE:
    {joined}
    """
)

# Sentence-BERT for embeddings
llm = SentenceTransformer("thenlper/gte-large")


# ==================================================================
# Helper functions
# ==================================================================

def one_sentence_summary(responses: List[str]) -> str:
    """Return a single-sentence Gemini summary for *responses*."""
    prompt = PROMPT_TMPL.format(joined="\n".join(f"- {r}" for r in responses))
    response = genai_client.models.generate_content(
        model=MODEL_NAME,
        contents=[prompt],
    )
    return response.text


# ==================================================================
# Main streaming pipeline (with live progress reporting)
# ==================================================================

def stream_process() -> None:
    """Stream-process EXTENDED_JSON → OUTPUT_JSON with minimal RAM and show progress."""

    # Ensure output directory exists
    OUTPUT_JSON.parent.mkdir(parents=True, exist_ok=True)

    start_ts = time.time()
    processed = 0

    with EXTENDED_JSON.open("rb") as src, OUTPUT_JSON.open("w", encoding="utf-8") as dst:
        # Open JSON array
        dst.write("[\n")
        first = True
        last = False

        # Stream top-level array members
        items = ijson.items(src, "item")
        tot = 1000 
        for idx, seq in enumerate(tqdm(items, desc="Summaries", unit="seq")):
            
            processed += 1
            if idx == tot - 1:
                last = True

            if "responses" not in seq:
                raise ValueError(f"Object {idx} missing 'responses' key")

            # === summarise & embed ===
            sentence = one_sentence_summary(seq["responses"])
            seq["summary"] = sentence
            seq["y"] = llm.encode(sentence, convert_to_numpy=True).tolist()
            seq["h"] = [[float(x) for x in row] for row in seq["embeddings"]]

            # --- drop the raw embeddings now that we've copied what we need ---
            seq.pop("embeddings", None)

            json.dump(seq, dst, ensure_ascii=False, indent=2)
            
            # Comma management inside output array
            if not first and not last:
                dst.write(",\n")
            first = False

            # ─── heartbeat ───
            if processed % PRINT_EVERY == 0:
                elapsed = time.time() - start_ts
                print(f"[{processed}] processed – elapsed {elapsed:.1f}s")

        # Close JSON array
        dst.write("\n]\n")

    # Final run summary
    elapsed = time.time() - start_ts
    print(f"\n[INFO] Finished – wrote {processed} summaries to {OUTPUT_JSON.resolve()}")
    print(f"[INFO] Total time: {elapsed:.1f}s | Avg/item: {elapsed / processed:.2f}s")


# ==================================================================
# Entrypoint
# ==================================================================

if __name__ == "__main__":
    stream_process()
