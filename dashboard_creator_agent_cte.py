from __future__ import annotations

import json
import os
import re
import textwrap
from pathlib import Path
from typing import List, Dict, Any, Optional, Union

import pandas as pd
from openai import OpenAI

from config import OPENAI_API_KEY, OPENAI_BASE_URL, OPENAI_TIMEOUT, LLM_MODEL as MODEL_NAME, LLM_TOP_P, LLM_PRESENCE_PENALTY, LLM_FREQUENCY_PENALTY, LLM_SAMPLING_TOP_K, LLM_REPETITION_PENALTY

DASHBOARD_FILE: str = "created_dashboard.py"

# Initialize client for local LLM / OpenAI-compatible server
client = OpenAI(api_key=OPENAI_API_KEY, base_url=OPENAI_BASE_URL, timeout=OPENAI_TIMEOUT)

# Load system prompt from static file
_BASE_DIR = Path(__file__).resolve().parent
_PROMPT_PATH = _BASE_DIR / "static" / "dashboard_creator_system_prompt.txt"
try:
    SYSTEM_PROMPT: str = _PROMPT_PATH.read_text(encoding="utf-8")
except Exception:
    SYSTEM_PROMPT = "You are a top Python developer and Streamlit dashboard designer. Generate a full file."


def _extract_inner_code(text: str) -> str:
    text = (text or "").strip()
    if text.startswith("```"):
        text = re.sub(r"^```[\w-]*\n", "", text)
        text = re.sub(r"\n```$", "", text)
    return text


def _sanitize_sql_in_triple_quoted_strings(code: str) -> str:
    """Remove stray semicolons before closing parenthesis inside SQL blocks.
    Applied only within triple-quoted strings to avoid touching Python code.
    """
    def _fix(match: re.Match) -> str:
        quote = match.group(1)
        body = match.group(2)
        body = re.sub(r";\s*\)", ")", body)  # fix CTE inner semicolons
        body = re.sub(r";\s*$", "", body)     # drop trailing semicolon at end of SQL
        return f"{quote}{body}{quote}"

    pattern = re.compile(r"(\"\"\"|''')(.*?)(\1)", re.S)
    return re.sub(pattern, lambda m: _fix(m), code)


def _ensure_list_sql(sql_or_list: Union[str, List[str]]) -> List[str]:
    if isinstance(sql_or_list, list):
        return [s for s in sql_or_list if isinstance(s, str) and s.strip()]
    return [sql_or_list] if isinstance(sql_or_list, str) and sql_or_list.strip() else []


def generate_dashboard_code(
    kpi_list: List[Dict[str, Any]],
    creation_sql: Union[str, List[str]],
    filtered_schema: List[Dict[str, Any]],
    model: str = MODEL_NAME,
    temperature: float = 0.2,
    max_tokens: int = 2000,
) -> str:
    schema_str = json.dumps(filtered_schema, indent=2)
    sql_list = _ensure_list_sql(creation_sql)

    # If multiple queries present, provide them as indexed blocks; otherwise keep single placeholder
    if len(sql_list) <= 1:
        sql_block = (sql_list[0] if sql_list else "").strip()
        prompt = SYSTEM_PROMPT.replace("{{creation_sql}}", sql_block).replace("{{table_schema_block}}", f"<table_schema>\n{schema_str}\n</table_schema>")
    else:
        combined = "\n\n".join([f"-- Query {i+1}\n{q.strip()}" for i, q in enumerate(sql_list)])
        prompt = SYSTEM_PROMPT.replace("{{creation_sql}}", combined).replace("{{table_schema_block}}", f"<table_schema>\n{schema_str}\n</table_schema>")

    messages = [
        {"role": "system", "content": prompt},
        {
            "role": "user",
            "content": (
                f"<kpis>\n{kpi_list}\n</kpis>\n\n"
                f"<creation_sql>\n{combined if len(sql_list) > 1 else sql_block}\n</creation_sql>"
            ),
        },
    ]

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
        model=model,
        messages=messages,
        temperature=temperature,
        max_tokens=max_tokens,
        **extra_kwargs,
    )

    code = _extract_inner_code(resp.choices[0].message.content)
    code = _sanitize_sql_in_triple_quoted_strings(code)
    return code


def write_dashboard_file(code_str: str, path: str | Path = DASHBOARD_FILE) -> Path:
    path = Path(path)
    path.write_text(code_str, encoding="utf-8")
    print(f"✅ Dashboard script written to {path.resolve()}")
    return path
