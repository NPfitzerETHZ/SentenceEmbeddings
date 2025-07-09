#!/usr/bin/env python
"""
Summarize sequences from multiple JSON files using Gemini and Sentence-BERT embeddings,
and merge them into a single JSON output file with minimal memory footprint.

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

# Input files
INPUT_FILES = [
    (Path("sequential_tasks/data/dataset_no_summary_flag_home.json"), "FLAG_HOME"),
    (Path("sequential_tasks/data/dataset_no_summary_defend.json"), "DEFEND")
]

# Output file
OUTPUT_JSON = Path("sequential_tasks/data/dataset_full.json")

MODEL_NAME = "gemini-2.0-flash"
TEMP = 0.2
PRINT_EVERY = 100

PROMPTS = {
    "FLAG_HOME": textwrap.dedent("""
        You are leading a team of robots in a capture the flag mission. Your team must find the flag and then return to base.
        Write ONE sentence summarising the sequence of tasks. Ignore repetitions. Do **not** add anything else. The task sequence might mention an area of interest for the flag location.
        TASK SEQUENCE:
        {joined}
    """),

    "DEFEND": textwrap.dedent("""
        You are leading a team of robots in a capture the flag mission. Your team must find the flag and then defend it.
        Write ONE sentence summarising the sequence of tasks. Ignore repetitions. Do **not** add anything else. The task sequence might mention an area of interest for the flag location.
        TASK SEQUENCE:
        {joined}
    """)
}

llm = SentenceTransformer("thenlper/gte-large")

MAX_PROMPT_CHARS = 24000  # safe limit for Gemini Flash 2.0 prompt input

def one_sentence_summary(responses: List[str], prompt_tmpl: str) -> str:
    
    bullet_responses = [f"- {r}" for r in responses]
    joined = "\n".join(bullet_responses)

    # Truncate if too long
    base_prompt = prompt_tmpl.format(joined="{joined}")
    max_joined_chars = MAX_PROMPT_CHARS - len(base_prompt.format(joined=""))
    
    if len(joined) > max_joined_chars:
        # Truncate from the start (keep latest instructions, which are likely most relevant)
        truncated = joined[-max_joined_chars:]
        # Ensure we don't cut in the middle of a bullet
        truncated = "\n".join(truncated.split("\n")[1:])
    else:
        truncated = joined

    prompt = prompt_tmpl.format(joined=truncated)
    print(prompt)
    response = genai_client.models.generate_content(
        model=MODEL_NAME,
        contents=[prompt],
    )
    return response.text

# Main streaming pipeline
def stream_process():
    OUTPUT_JSON.parent.mkdir(parents=True, exist_ok=True)

    start_ts = time.time()
    processed = 0

    with OUTPUT_JSON.open("w", encoding="utf-8") as dst:
        dst.write("[\n")
        first_entry = True

        for input_file, prompt_key in INPUT_FILES:
            prompt_template = PROMPTS[prompt_key]

            with input_file.open("rb") as src:
                items = ijson.items(src, "item")

                for seq in tqdm(items, desc=f"Summarizing {input_file.name}", unit="seq"):
                    if "responses" not in seq:
                        raise ValueError(f"Object missing 'responses' key")

                    sentence = one_sentence_summary(seq["responses"], prompt_template)
                    seq["summary"] = sentence
                    seq["y"] = llm.encode(sentence, convert_to_numpy=True).tolist()
                    seq["h"] = [[float(x) for x in row] for row in seq["embeddings"]]
                    seq.pop("embeddings", None)

                    if not first_entry:
                        dst.write(",\n")
                    else:
                        first_entry = False

                    json.dump(seq, dst, ensure_ascii=False, indent=2)

                    processed += 1
                    if processed % PRINT_EVERY == 0:
                        elapsed = time.time() - start_ts
                        print(f"[{processed}] processed – elapsed {elapsed:.1f}s")

        dst.write("\n]\n")

    elapsed = time.time() - start_ts
    print(f"\n[INFO] Finished – wrote {processed} summaries to {OUTPUT_JSON.resolve()}")
    print(f"[INFO] Total time: {elapsed:.1f}s | Avg/item: {elapsed / processed:.2f}s")

# Entrypoint
if __name__ == "__main__":
    stream_process()