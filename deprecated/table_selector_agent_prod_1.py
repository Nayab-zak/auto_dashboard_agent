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
from config import OPENAI_API_KEY, EMBED_MODEL, LLM_MODEL, SIM_CUTOFF, TOP_K, OPENAI_BASE_URL, OPENAI_TIMEOUT

TABLE_SELECTOR_SYSTEM_PROMPT_PATH = "static/table_selector_system_prompt.txt"
with open(TABLE_SELECTOR_SYSTEM_PROMPT_PATH, "r") as f:
    SYSTEM_PROMPT = f.read()

# ------------------------------------------------------------------------------
# Configuration
# ------------------------------------------------------------------------------
# DB_PATH = "path/to/port_operations.db"
SCHEMA_METADATA_JSON = "agent1_schema_metadata_with_samples.json"
# Replace 'your_openai_api_key' with your actual API key
os.environ['OPENAI_API_KEY'] = OPENAI_API_KEY

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
client = OpenAI(api_key=OPENAI_API_KEY, base_url=OPENAI_BASE_URL or None)
JSON_MODE = bool(OPENAI_BASE_URL)

# Robust JSON helpers (for local models)
import re
_fence_re = re.compile(r"```[\w-]*\s*\n([\s\S]*?)\n?```$")

def _strip_markdown_fence(text: str) -> str:
    t = (text or "").strip()
    if t.startswith("```") and t.endswith("```"):
        m = _fence_re.match(t)
        if m:
            return m.group(1).strip()
    return text or ""


def _extract_json_dict(text: str):
    t = _strip_markdown_fence(text)
    try:
        return json.loads(t)
    except Exception:
        pass
    if "{" in t and "}" in t:
        s, e = t.find("{"), t.rfind("}")
        if e > s:
            snippet = t[s:e+1]
            try:
                return json.loads(snippet)
            except Exception:
                pass
    try:
        return json.loads(t.replace("'", '"'))
    except Exception:
        return {}


def _cosine(a: np.ndarray, b: np.ndarray) -> float:
    return float(np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b)))


def _embed(texts: List[str]) -> List[np.ndarray]:
    try:
        res = client.embeddings.create(model=EMBED_MODEL, input=texts, timeout=OPENAI_TIMEOUT)
        return [np.array(e.embedding, dtype="float32") for e in res.data]
    except Exception as e:
        print(f"[Table Selector] Embeddings unavailable or failed: {e}")
        # Embeddings may be unavailable on some OpenAI-compatible servers; return empty to trigger LLM fallback
        return []


tbl_desc_blocks = [
    f"{tbl}: {TABLE_META[tbl]['desc']}. Columns: {', '.join(TABLE_META[tbl]['cols'])}"
    for tbl in TABLE_NAMES
]
_embeds = _embed(tbl_desc_blocks)
TABLE_EMB: List[np.ndarray] = _embeds if _embeds and len(_embeds) == len(tbl_desc_blocks) else []

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

# ------------------------------------------------------------------------------
# LLM Table Selector
# ------------------------------------------------------------------------------
def _llm_select_table(question: str, sims: List[float], *, max_retries: int = 2) -> List[str]:
    messages: List[dict] = [{"role": "system", "content": SYSTEM_PROMPT}]

    if JSON_MODE:
        # Few-shot with explicit JSON responses (no tools)
        for q, tbls in EXAMPLES:
            messages.append({"role": "user", "content": q})
            messages.append({"role": "assistant", "content": json.dumps({"table": tbls})})
    else:
        # Tool-based few-shot
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
    user_content = (
        f"User question: {question}\n\nHere are the candidate tables:\n{tbl_block}"
    )
    if JSON_MODE:
        user_content += "\n\nReturn ONLY a compact JSON object like {\"table\": [\"tbl1\", ...]} with valid table names. No prose."
    messages.append({"role": "user", "content": user_content})

    for attempt in range(max_retries + 1):
        if JSON_MODE:
            resp: ChatCompletion = client.chat.completions.create(
                model=LLM_MODEL,
                temperature=0.0,
                messages=messages,
                timeout=OPENAI_TIMEOUT,
            )
            try:
                choices = getattr(resp, "choices", []) or []
                if not choices:
                    print(f"[Table Selector] Attempt {attempt}: empty choices from LLM")
                    raise RuntimeError("Empty LLM response")
                content = choices[0].message.content or "{}"
                tables = json.loads(content).get("table", [])
                valid_tables = [t for t in tables if t in TABLE_NAMES]
                if valid_tables:
                    return valid_tables
            except Exception:
                pass
        else:
            resp: ChatCompletion = client.chat.completions.create(
                model=LLM_MODEL,
                temperature=0.0,
                messages=messages,
                tools=[SELECT_TABLE_TOOL],
                tool_choice={"type": "function", "function": {"name": "select_table"}},
                timeout=OPENAI_TIMEOUT,
            )
            try:
                choices = getattr(resp, "choices", []) or []
                if not choices:
                    print(f"[Table Selector] Attempt {attempt}: empty choices from LLM (tool mode)")
                    raise RuntimeError("Empty LLM response")
                if choices[0].message.tool_calls:
                    tool_call = choices[0].message.tool_calls[0]
                    tables = json.loads(tool_call.function.arguments)["table"]
                else:
                    content = choices[0].message.content or "{}"
                    tables = json.loads(content).get("table", [])
                valid_tables = [t for t in tables if t in TABLE_NAMES]
                if valid_tables:
                    return valid_tables
            except Exception:
                pass

        if attempt < max_retries:
            if JSON_MODE:
                messages.append({
                    "role": "assistant",
                    "content": (
                        f"Invalid. Return JSON only with key 'table' and values from: {TABLE_NAMES}"
                    ),
                })
            else:
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
    if TABLE_EMB:
        q_emb = _embed([question])
        if q_emb:
            q_emb = q_emb[0]
            sims = [_cosine(q_emb, te) for te in TABLE_EMB]
            selected = [TABLE_NAMES[i] for i, sim in enumerate(sims) if sim >= SIM_CUTOFF]
            if selected and not force_llm:
                return selected
        else:
            sims = [0.0 for _ in TABLE_NAMES]
    else:
        sims = [0.0 for _ in TABLE_NAMES]
    return _llm_select_table(question, sims)


def recommend_table(question: str) -> List[str]:
    return choose_table(question)
