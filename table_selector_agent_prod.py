"""
===============================================================================
 table_selector_agent.py  ▸  Port‑Operations Table Router (Agent 1)
===============================================================================
A self‑contained utility that maps a free‑form user question to one or more
SQLite tables from a production `port_operations.db` using a two-stage approach:

1. Embedding router – cosine similarity via `text-embedding-3-small`.
2. LLM fallback – structured few-shot function-calling with `gpt-4o`.

The final schema is loaded from a JSON file that contains descriptions,
column info, joins, and sample data for each production table.

Author: Auto-generated with full metadata support.
"""

import os
import json
import numpy as np
from typing import List
from openai import OpenAI
from openai.types.chat import ChatCompletion
from config import OPENAI_API_KEY

# ------------------------------------------------------------------------------
# Configuration
# ------------------------------------------------------------------------------
# DB_PATH = "path/to/port_operations.db"
# Use relative path for schema metadata JSON
SCHEMA_METADATA_JSON = os.path.join(os.path.dirname(__file__), "agent1_schema_metadata_with_samples.json")
EMBED_MODEL = os.getenv("EMBED_MODEL", "text-embedding-3-small")
LLM_MODEL = os.getenv("LLM_MODEL", "gpt-4o")
SIM_CUTOFF = float(os.getenv("SIM_CUTOFF", 0.80))
TOP_K = int(os.getenv("TOP_K", 3))
# ------------------------------------------------------------------------------
# Load table schema metadata
# ------------------------------------------------------------------------------
with open(SCHEMA_METADATA_JSON, "r") as f:
    schema_metadata = json.load(f)

TABLE_NAMES = [table["table_name"] for table in schema_metadata]
TABLE_META = {
    table["table_name"]: {
        "desc": table.get("column_info", {}).get("ROTN", {}).get("description", "(no description provided)"),
        "cols": list(table.get("column_info", {}).keys())
    } for table in schema_metadata
}

# ------------------------------------------------------------------------------
# Embedding and LLM setup
# ------------------------------------------------------------------------------
client = OpenAI(api_key=OPENAI_API_KEY)

def _cosine(a: np.ndarray, b: np.ndarray) -> float:
    return float(np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b)))

def _embed(texts: List[str]) -> List[np.ndarray]:
    res = client.embeddings.create(model=EMBED_MODEL, input=texts)
    return [np.array(e.embedding, dtype="float32") for e in res.data]

TABLE_EMB: List[np.ndarray] = _embed([
    f"{tbl}: {TABLE_META[tbl]['desc']}. Columns: {', '.join(TABLE_META[tbl]['cols'])}"
    for tbl in TABLE_NAMES
])

# ------------------------------------------------------------------------------
# Few-shot Examples
# ------------------------------------------------------------------------------
EXAMPLES = [
    ("Show all vessel rotations where discharge, load and marine are completed.",
     ["DPW_DL.Voyages"]),
    ("Compare TEUs handled for 20ft and 40ft containers in Jebel Ali.",
     ["DPW_DL.EDW_VOYAGE_BOX_STAT"]),
    ("List shipping line names for BOX_LINE_IDs present in voyage box stats.",
     ["DPW_DL.EDW_VOYAGE_BOX_STAT", "DPW_DL.DM_LINES"]),
    ("Which vessel lines carried the highest TEUs in 2023 by route code?",
     ["DPW_DL.Voyages", "DPW_DL.EDW_VOYAGE_BOX_STAT"]),
    ("Give me completed voyages along with TEUs handled and shipping line names.",
     ["DPW_DL.Voyages", "DPW_DL.EDW_VOYAGE_BOX_STAT", "DPW_DL.DM_LINES"]),
    ("List voyages where lay_by_vessel is Y and show related terminal-wise TEU activity.",
     ["DPW_DL.Voyages", "DPW_DL.EDW_VOYAGE_BOX_STAT"])
]

# ------------------------------------------------------------------------------
# Function Tool Format for LLM Routing
# ------------------------------------------------------------------------------
SELECT_TABLE_TOOL = {
    "type": "function",
    "function": {
        "name": "select_table",
        "description": "Return one or more SQLite tables that best answer the user's question.",
        "parameters": {
            "type": "object",
            "properties": {
                "table": {
                    "type": "array",
                    "items": {
                        "type": "string",
                        "enum": TABLE_NAMES
                    },
                    "description": "List of one or more relevant table names"
                }
            },
            "required": ["table"],
        },
    },
}

SYSTEM_PROMPT = (
    "You are a strict SQL metadata router. "
    "Return one or more table names from the list based on the user query. "
    "If none are relevant, return an empty list. Reply *only* using the function schema."
)

# ------------------------------------------------------------------------------
# LLM Table Selector
# ------------------------------------------------------------------------------
def _llm_select_table(question: str, sims: List[float], *, max_retries: int = 2) -> List[str]:
    messages: List[dict] = [{"role": "system", "content": SYSTEM_PROMPT}]
    for i, (q, tbls) in enumerate(EXAMPLES):
        call_id = f"example_call_{i}"
        messages.extend([
            {"role": "user", "content": q},
            {
                "role": "assistant",
                "content": None,
                "tool_calls": [{
                    "id": call_id,
                    "type": "function",
                    "function": {
                        "name": "select_table",
                        "arguments": json.dumps({"table": tbls}),
                    },
                }],
            },
            {
                "role": "tool",
                "tool_call_id": call_id,
                "name": "select_table",
                "content": json.dumps({"table": tbls}),
            },
        ])

    top_idx = np.argsort(sims)[::-1][:TOP_K]
    tbl_block = "\n".join(
        f"- **{TABLE_NAMES[i]}**: {TABLE_META[TABLE_NAMES[i]]['desc']}"
        for i in top_idx
    )
    messages.append({
        "role": "user",
        "content": (
            f"User question: {question}\n\nHere are the candidate tables:\n{tbl_block}"
        ),
    })

    for attempt in range(max_retries + 1):
        resp: ChatCompletion = client.chat.completions.create(
            model=LLM_MODEL,
            temperature=0.0,
            messages=messages,
            tools=[SELECT_TABLE_TOOL],
            tool_choice={"type": "function", "function": {"name": "select_table"}},
        )
        tool_call = resp.choices[0].message.tool_calls[0]
        try:
            tables = json.loads(tool_call.function.arguments)["table"]
            valid_tables = [t for t in tables if t in TABLE_NAMES]
            return valid_tables
        except Exception:
            pass

        if attempt < max_retries:
            messages.append({
                "role": "assistant",
                "content": (
                    f"Previous response invalid. Return a list of valid tables: {TABLE_NAMES}"
                ),
            })

    raise RuntimeError("LLM failed to produce valid table list after retries.")

# ------------------------------------------------------------------------------
# API Entry Point
# ------------------------------------------------------------------------------
def choose_table(question: str, *, force_llm: bool = False) -> List[str]:
    q_emb, = _embed([question])
    sims = [_cosine(q_emb, te) for te in TABLE_EMB]
    selected = [TABLE_NAMES[i] for i, sim in enumerate(sims) if sim >= SIM_CUTOFF]
    if selected and not force_llm:
        return selected
    return _llm_select_table(question, sims)

def recommend_table(question: str) -> List[str]:
    return choose_table(question)
