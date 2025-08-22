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
from typing import List, Optional
from pathlib import Path
from openai import OpenAI
from openai.types.chat import ChatCompletion
import re

# ------------------------------------------------------------------------------
# Configuration
# ------------------------------------------------------------------------------
from config import (
    OPENAI_API_KEY,
    OPENAI_BASE_URL,
    OPENAI_TIMEOUT,
    EMBED_MODEL,
    LLM_MODEL,
    SIM_CUTOFF,
    TOP_K,
    USE_LOCAL_EMBEDDINGS,
    LLM_TOP_P,
    LLM_PRESENCE_PENALTY,
    LLM_FREQUENCY_PENALTY,
    LLM_SAMPLING_TOP_K,
    LLM_REPETITION_PENALTY,
)

BASE_DIR = Path(__file__).resolve().parent
SCHEMA_METADATA_JSON = str(BASE_DIR / "agent1_schema_metadata_with_samples.json")

# ------------------------------------------------------------------------------
# Load table schema metadata
# ------------------------------------------------------------------------------
with open(SCHEMA_METADATA_JSON, "r", encoding="utf-8") as f:
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
client = OpenAI(api_key=OPENAI_API_KEY, base_url=OPENAI_BASE_URL, timeout=OPENAI_TIMEOUT)


def _cosine(a: np.ndarray, b: np.ndarray) -> float:
    return float(np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b)))


def _embed_openai(texts: List[str]) -> List[np.ndarray]:
    res = client.embeddings.create(model=EMBED_MODEL, input=texts)
    return [np.array(e.embedding, dtype="float32") for e in res.data]


def _embed_local_st(texts: List[str]) -> List[np.ndarray]:
    # Lazy import to avoid heavy deps at startup
    from sentence_transformers import SentenceTransformer
    # If EMBED_MODEL looks like an OpenAI name, pick a sane local default
    local_model = EMBED_MODEL
    if "text-embedding" in (EMBED_MODEL or ""):
        local_model = "sentence-transformers/all-MiniLM-L6-v2"
    model = SentenceTransformer(local_model)
    emb = model.encode(texts, normalize_embeddings=True, convert_to_numpy=True)
    # Ensure list[np.ndarray]
    return [np.array(v, dtype="float32") for v in emb]


def _embed(texts: List[str]) -> List[np.ndarray]:
    """Always use local sentence-transformers for embeddings (no API call)."""
    return _embed_local_st(texts)


# Lazy cache of table embeddings to avoid failing at import time
_TABLE_EMB: Optional[List[np.ndarray]] = None


def _table_texts() -> List[str]:
    return [
        f"{tbl}: {TABLE_META[tbl]['desc']}. Columns: {', '.join(TABLE_META[tbl]['cols'])}"
        for tbl in TABLE_NAMES
    ]


def _ensure_table_embeddings() -> List[np.ndarray]:
    global _TABLE_EMB
    if _TABLE_EMB is None:
        _TABLE_EMB = _embed(_table_texts())
    return _TABLE_EMB

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

# Load system prompt from static file
_DEF_PROMPT_PATH = BASE_DIR / "static" / "table_selector_system_prompt.txt"
try:
    SYSTEM_PROMPT = _DEF_PROMPT_PATH.read_text(encoding="utf-8")
except Exception:
    SYSTEM_PROMPT = (
        "You are a strict SQL metadata router. "
        "Return one or more table names from the list based on the user query. "
        "If none are relevant, return an empty list. Reply only using the function schema."
    )

# ------------------------------------------------------------------------------
# LLM Table Selector
# ------------------------------------------------------------------------------

def _extract_json_object(text: str) -> str:
    text = (text or "").strip()
    if text.startswith("```"):
        text = re.sub(r"^```[\w-]*\n", "", text)
        text = re.sub(r"\n```$", "", text)
    start = text.find("{")
    end = text.rfind("}")
    return text[start:end+1] if start != -1 and end != -1 and end > start else text


def _extract_tables_from_text(text: str) -> List[str]:
    """Heuristic: scan the model output for any known table names.
    Returns unique tables in order of appearance.
    """
    found: List[str] = []
    for name in TABLE_NAMES:
        if re.search(rf"\b{re.escape(name)}\b", text):
            if name not in found:
                found.append(name)
    return found


def _llm_select_table(question: str, sims: List[float], *, max_retries: int = 2) -> List[str]:
    instruct = (
        SYSTEM_PROMPT.strip()
        + "\n\nReturn ONLY a single JSON object with this exact shape: {\"table\": [<valid table names>]}.\n"
        + f"Valid table names: {TABLE_NAMES}\n"
    )
    messages: List[dict] = [{"role": "system", "content": instruct}]

    # Few-shot as JSON examples (no tool calls)
    for i, (q, tbls) in enumerate(EXAMPLES):
        messages.extend([
            {"role": "user", "content": q},
            {"role": "assistant", "content": json.dumps({"table": tbls})},
        ])

    top_idx = np.argsort(sims)[::-1][:TOP_K]
    tbl_block = "\n".join(
        f"- {TABLE_NAMES[i]}: {TABLE_META[TABLE_NAMES[i]]['desc']}"
        for i in top_idx
    )
    messages.append({
        "role": "user",
        "content": (
            f"User question: {question}\n\nHere are the candidate tables:\n{tbl_block}\n\n"
            "Choose from the valid table names only and answer with JSON."
        ),
    })

    for attempt in range(max_retries + 1):
        extra_kwargs = {
            "top_p": LLM_TOP_P,
            "presence_penalty": LLM_PRESENCE_PENALTY,
            "frequency_penalty": LLM_FREQUENCY_PENALTY,
        }
        if LLM_SAMPLING_TOP_K is not None:
            extra_kwargs["top_k"] = LLM_SAMPLING_TOP_K
        if LLM_REPETITION_PENALTY is not None:
            extra_kwargs["repetition_penalty"] = LLM_REPETITION_PENALTY

        resp = client.chat.completions.create(
            model=LLM_MODEL,
            temperature=0.0,
            messages=messages,
            **extra_kwargs,
        )
        content = (resp.choices[0].message.content or "").strip()
        # Debug print to trace failures
        print("[TableSelector] Raw LLM content:", content[:500])
        try:
            parsed = json.loads(_extract_json_object(content))
            tables = parsed.get("table") or parsed.get("tables") or parsed.get("table_names") or []
            valid_tables = [t for t in tables if t in TABLE_NAMES]
            if valid_tables:
                return valid_tables
        except Exception:
            pass

        # Heuristic extraction of table names from free-form text
        extracted = _extract_tables_from_text(content)
        if extracted:
            return extracted

        if attempt < max_retries:
            messages.append({
                "role": "assistant",
                "content": (
                    "Invalid format. Return only JSON like {\"table\": [\"T1\", \"T2\"]} using valid names."
                ),
            })

    # Final fallback: return top-similarity candidates to avoid pipeline failure
    fallback = [TABLE_NAMES[i] for i in top_idx[: max(1, min(TOP_K, 2))]]
    print("[TableSelector] Falling back to similarity tables:", fallback)
    return fallback

# ------------------------------------------------------------------------------
# API Entry Point
# ------------------------------------------------------------------------------

def choose_table(question: str, *, force_llm: bool = False) -> List[str]:
    table_emb = _ensure_table_embeddings()
    q_emb, = _embed([question])
    sims = [_cosine(q_emb, te) for te in table_emb]
    selected = [TABLE_NAMES[i] for i, sim in enumerate(sims) if sim >= SIM_CUTOFF]
    if selected and not force_llm:
        return selected
    # If nothing high-confidence, still proceed; the LLM method has robust fallbacks
    return _llm_select_table(question, sims)


def recommend_table(question: str) -> List[str]:
    return choose_table(question)

