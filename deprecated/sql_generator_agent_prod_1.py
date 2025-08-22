from __future__ import annotations

import json
import os
import re
from typing import Any, Dict, List, Tuple, Union

import pandas as pd
from openai import OpenAI
from openai.types.chat.chat_completion_message_tool_call import (
    ChatCompletionMessageToolCall,
)
from config import OPENAI_API_KEY, LLM_MODEL, OPENAI_BASE_URL, OPENAI_TIMEOUT

SCHEMA_JSON_PATH = "./agent1_schema_metadata_with_samples.json"
MAX_RETRIES = 2

client = OpenAI(api_key=OPENAI_API_KEY, base_url=OPENAI_BASE_URL or None)
JSON_MODE = bool(OPENAI_BASE_URL)
LLM_MODEL = LLM_MODEL

SQL_GENERATOR_SYSTEM_PROMPT_PATH = "static/sql_generator_system_prompt.txt"
with open(SQL_GENERATOR_SYSTEM_PROMPT_PATH, "r") as f:
    SYSTEM_PROMPT = f.read()

# --- JSON-mode helpers (mirror KPI planner robustness) ---
_def_fence_re = re.compile(r"```[\w-]*\s*\n([\s\S]*?)\n?```$")

def _strip_markdown_fence(text: str) -> str:
    t = (text or "").strip()
    if t.startswith("```") and t.endswith("```"):
        m = _def_fence_re.match(t)
        if m:
            return m.group(1).strip()
    return text or ""


def _extract_json_dict(text: str) -> Dict[str, Any]:
    t = _strip_markdown_fence(text)
    # direct
    try:
        return json.loads(t)
    except Exception:
        pass
    # find outermost braces
    if "{" in t and "}" in t:
        start = t.find("{")
        end = t.rfind("}")
        if end > start:
            snippet = t[start : end + 1]
            try:
                return json.loads(snippet)
            except Exception:
                pass
    # single-quote fallback
    try:
        return json.loads(t.replace("'", '"'))
    except Exception:
        return {}

# --- Deterministic SQL fallback ---
_alias_defaults = {
    "DPW_DL.EDW_VOYAGE_BOX_STAT": "b",
    "DPW_DL.Voyages": "v",
    "DPW_DL.DM_LINES": "l",
}

def _fallback_sql(tables: List[str], col_map: Dict[str, List[str]]) -> str:
    # Choose base table: prefer BOX_STAT, else Voyages, else first
    base = None
    if "DPW_DL.EDW_VOYAGE_BOX_STAT" in tables:
        base = "DPW_DL.EDW_VOYAGE_BOX_STAT"
    elif "DPW_DL.Voyages" in tables:
        base = "DPW_DL.Voyages"
    else:
        base = tables[0]

    aliases: Dict[str, str] = {}
    used = set()
    for t in tables:
        a = _alias_defaults.get(t)
        if not a:
            # generate t1, t2...
            i = 1
            while f"t{i}" in used:
                i += 1
            a = f"t{i}"
        aliases[t] = a
        used.add(a)

    select_cols: List[str] = []
    for t in tables:
        a = aliases[t]
        for c in col_map.get(t, []):
            if t == "DPW_DL.EDW_VOYAGE_BOX_STAT" and c.upper() == "YEAR_MTH":
                # include raw and derived date
                select_cols.append(f"{a}.YEAR_MTH AS {a}_YEAR_MTH")
                select_cols.append(
                    f"TO_DATE(CAST({a}.YEAR_MTH AS VARCHAR(6)) || '01', 'YYYYMMDD') AS {a}_YEAR_MTH_DATE"
                )
            elif t == "DPW_DL.EDW_VOYAGE_BOX_STAT" and c.upper() == "TERMINAL_ID":
                select_cols.append(f"('T' || {a}.TERMINAL_ID) AS {a}_TERMINAL_ID")
            else:
                select_cols.append(f"{a}.{c} AS {a}_{c}")

    if not select_cols:
        # safety: at least project ROTN when present
        a = aliases[base]
        select_cols = [f"{a}.*"]

    base_alias = aliases[base]
    from_clause = f"FROM {base} {base_alias}"

    # add known joins
    joins: List[str] = []
    if base == "DPW_DL.EDW_VOYAGE_BOX_STAT" and "DPW_DL.Voyages" in tables:
        joins.append(
            f"LEFT JOIN DPW_DL.Voyages {aliases['DPW_DL.Voyages']} ON {aliases['DPW_DL.EDW_VOYAGE_BOX_STAT']}.ROTN = {aliases['DPW_DL.Voyages']}.ROTN"
        )
    if base == "DPW_DL.Voyages" and "DPW_DL.EDW_VOYAGE_BOX_STAT" in tables:
        joins.append(
            f"LEFT JOIN DPW_DL.EDW_VOYAGE_BOX_STAT {aliases['DPW_DL.EDW_VOYAGE_BOX_STAT']} ON {aliases['DPW_DL.Voyages']}.ROTN = {aliases['DPW_DL.EDW_VOYAGE_BOX_STAT']}.ROTN"
        )
    # DM_LINES join through BOX_STAT when available
    if "DPW_DL.DM_LINES" in tables:
        if "DPW_DL.EDW_VOYAGE_BOX_STAT" in tables:
            joins.append(
                f"LEFT JOIN DPW_DL.DM_LINES {aliases['DPW_DL.DM_LINES']} ON {aliases['DPW_DL.EDW_VOYAGE_BOX_STAT']}.BOX_LINE_ID = {aliases['DPW_DL.DM_LINES']}.LINE_ID"
            )

    sql = "SELECT\n  " + ",\n  ".join(select_cols) + f"\n{from_clause}"
    if joins:
        sql += "\n" + "\n".join(joins)
    return sql


def load_schema() -> List[Dict[str, Any]]:
    with open(SCHEMA_JSON_PATH, "r") as f:
        return json.load(f)


def get_table_columns_from_json(schema: List[Dict[str, Any]]) -> Dict[str, List[str]]:
    return {tbl["table_name"]: list(tbl["column_info"].keys()) for tbl in schema}


def _valid(sql: str, tables: List[str], col_map: Dict[str, List[str]]) -> bool:
    if re.search(r"\b(insert|update|delete|drop|alter|truncate)\b", sql, re.I):
        return False
    if not sql.strip().upper().startswith("SELECT"):
        return False
    for table in tables:
        if not re.search(rf"\b{re.escape(table)}\b", sql, re.I):
            return False
    for table, cols in col_map.items():
        for col in cols:
            if not re.search(rf"\b{re.escape(col)}\b", sql, re.I):
                return False
    return True

GEN_SQL_TOOL = {
    "type": "function",
    "function": {
        "name": "generate_sql",
        "description": (
            "Return one or two SQLite SELECT queries that fetch all required columns "
            "from the KPI list. Use appropriate joins based on common columns across tables. "
            "Only return multiple queries if absolutely necessary."
        ),
        "parameters": {
            "type": "object",
            "properties": {
                "sql": {
                    "type": ["string", "array"],
                    "items": {"type": "string"}
                }
            },
            "required": ["sql"],
        },
    },
}


def extract_required_tables_and_columns(kpi_list: List[Dict[str, Any]]) -> Tuple[List[str], Dict[str, List[str]]]:
    table_column_map: Dict[str, set] = {}
    for kpi in kpi_list:
        for table in kpi.get("table_names", []):
            if table not in table_column_map:
                table_column_map[table] = set()
        for table, columns in kpi.get("required_columns", {}).items():
            if table not in table_column_map:
                table_column_map[table] = set()
            table_column_map[table].update(columns)
    return list(table_column_map.keys()), {k: list(v) for k, v in table_column_map.items()}


def _generate_sql_string(tables: List[str], col_map: Dict[str, List[str]], kpi_list: List[Dict[str, Any]], filtered_schema: List[Dict[str, Any]]) -> Union[str, List[str]]:
    schema_str = json.dumps(filtered_schema, indent=2)
    system_prompt_with_schema = SYSTEM_PROMPT.replace("{{table_schema_block}}", schema_str)
    user_msg = (
        f"Tables: {tables}\n"
        f"all_required_columns_by_table: {json.dumps(col_map, ensure_ascii=False)}\n"
        f"KPI list (JSON): {json.dumps(kpi_list, ensure_ascii=False)}"
    )

    messages = [
        {"role": "system", "content": system_prompt_with_schema},
        {"role": "user", "content": user_msg},
    ]

    for attempt in range(MAX_RETRIES + 1):
        resp = None
        try:
            resp = client.chat.completions.create(
                model=LLM_MODEL,
                temperature=0.0,
                messages=messages,
                timeout=OPENAI_TIMEOUT,
                **({} if JSON_MODE else {"tools": [GEN_SQL_TOOL], "tool_choice": {"type": "function", "function": {"name": "generate_sql"}}}),
            )
        except Exception as e:
            print(f"[SQL Gen] Attempt {attempt} request failed: {e}")
            resp = None
        try:
            choices = getattr(resp, "choices", []) or [] if resp is not None else []
            if not choices:
                print(f"[SQL Gen] Attempt {attempt}: empty choices from LLM")
                raise RuntimeError("Empty LLM response")
            
            if JSON_MODE:
                content = (choices[0].message.content or "").strip()
                print(f"[SQL Gen] Attempt {attempt} raw content:\n{content}\n")
                parsed = _extract_json_dict(content)
                sql_result = parsed.get("sql", [])
            else:
                call: ChatCompletionMessageToolCall | None = None
                if choices[0].message.tool_calls:
                    call = choices[0].message.tool_calls[0]
                    print(f"[SQL Gen] Attempt {attempt} tool args:\n{call.function.arguments}\n")
                    sql_result = json.loads(call.function.arguments)["sql"]
                else:
                    content = (choices[0].message.content or "").strip()
                    print(f"[SQL Gen] Attempt {attempt} fallback content:\n{content}\n")
                    parsed = _extract_json_dict(content)
                    sql_result = parsed.get("sql", [])
            if isinstance(sql_result, str):
                sql_result = [sql_result]
        except Exception:
            sql_result = []

        # Only accept if we have at least one valid SELECT
        if isinstance(sql_result, list) and sql_result and all(_valid(q, tables, col_map) for q in sql_result):
            return sql_result if len(sql_result) > 1 else sql_result[0]

        if attempt < MAX_RETRIES:
            messages.append(
                {
                    "role": "user",
                    "content": (
                        "Return JSON only with key 'sql'. Provide SELECT-only query/queries that reference all tables "
                        f"{tables} and include all required columns {col_map}."
                    ),
                }
            )
        else:
            # final debug hint
            if resp is not None:
                print("[SQL Gen] Final raw content:", getattr(resp.choices[0].message, 'content', ''))
            else:
                print("[SQL Gen] No final content due to repeated request failures.")

    # Deterministic fallback
    fb = _fallback_sql(tables, col_map)
    if _valid(fb, tables, col_map):
        print("[SQL Gen] Using deterministic fallback SQL.")
        return fb

    raise RuntimeError("LLM could not produce a valid SQL query.")


def generate_sql(kpi_list: List[Dict[str, Any]], filtered_schema: List[Dict[str, Any]]) -> Union[str, List[str]]:
    tables, col_map = extract_required_tables_and_columns(kpi_list)
    return _generate_sql_string(tables, col_map, kpi_list, filtered_schema)
