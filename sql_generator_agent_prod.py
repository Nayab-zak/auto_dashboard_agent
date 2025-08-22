from __future__ import annotations

import json
import os
import re
from typing import Any, Dict, List, Tuple, Union

import pandas as pd
from pathlib import Path
from openai import OpenAI
import pickle

from config import OPENAI_API_KEY, OPENAI_BASE_URL, OPENAI_TIMEOUT, LLM_MODEL, LLM_TOP_P, LLM_PRESENCE_PENALTY, LLM_FREQUENCY_PENALTY, LLM_SAMPLING_TOP_K, LLM_REPETITION_PENALTY, DEBUG_DIR

_BASE_DIR = Path(__file__).resolve().parent
SCHEMA_JSON_PATH = str(_BASE_DIR / "agent1_schema_metadata_with_samples.json")
MAX_RETRIES = 2

# Initialize client for local LLM or OpenAI-compatible server
client = OpenAI(api_key=OPENAI_API_KEY, base_url=OPENAI_BASE_URL, timeout=OPENAI_TIMEOUT)


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
    # Ensure each required table is referenced
    for table in tables:
        if not re.search(rf"\b{re.escape(table)}\b", sql, re.I):
            return False
    # Be lenient on columns: at least one required column per table must appear
    for table, cols in col_map.items():
        if cols:
            if not any(re.search(rf"\b{re.escape(col)}\b", sql, re.I) for col in cols):
                return False
    # Basic alias mismatch: if SELECT lists aliased expression, ensure outer references match alias or raw name
    return True


def _extract_select_aliases(sql: str) -> List[str]:
    # crude: capture aliases in top-level SELECT ... AS alias, alias without AS, and CTE column list
    aliases: List[str] = []
    # FROM start
    m = re.search(r"^\s*select\s+(.*?)\s+from\s", sql, re.I | re.S)
    if m:
        sel = m.group(1)
        # split on commas not within parentheses
        parts = re.split(r",(?=(?:[^()]*\([^()]*\))*[^()]*$)", sel)
        for p in parts:
            # ... AS alias
            as_m = re.search(r"\bas\s+([\w\$#]+)\b", p, re.I)
            if as_m:
                aliases.append(as_m.group(1))
                continue
            # ... expr alias (without AS) -> last token if not function call
            plain_m = re.findall(r"\b([A-Za-z_][\w$#]*)\b", p)
            if plain_m:
                aliases.append(plain_m[-1])
    # CTE column list: with cte(alias1, alias2) as (
    for cte_cols in re.findall(r"with\s+[\w$#]+\s*\(([^)]*)\)\s*as\s*\(", sql, re.I):
        aliases.extend([c.strip() for c in cte_cols.split(',') if c.strip()])
    return list(dict.fromkeys(aliases))


def _outer_refs_after_cte(sql: str) -> List[str]:
    # capture select of outer query if there is a closing ) followed by select
    m = re.search(r"\)\s*select\s+(.*?)\s+from\s", sql, re.I | re.S)
    if not m:
        return []
    sel = m.group(1)
    tokens = re.findall(r"\b([A-Za-z_][\w$#]*)\b", sel)
    return list(dict.fromkeys(tokens))


def _alias_mismatch_notes(sql: str) -> str:
    aliases = _extract_select_aliases(sql)
    outer = _outer_refs_after_cte(sql)
    if not outer:
        return ""
    unknown = [t for t in outer if t.upper() not in {"SELECT", "AS", "DISTINCT"} and t not in aliases]
    if unknown:
        return (
            "Outer query references columns not present in CTE output: " + ", ".join(unknown) + ". "
            "Either add these names to the inner SELECT list or reference the actual alias names."
        )
    return ""


def _missing_reasons(sql_list: List[str], tables: List[str], col_map: Dict[str, List[str]]) -> str:
    reasons = []
    if not sql_list:
        return "Model returned no JSON/sql payload."
    for idx, sql in enumerate(sql_list):
        miss_tbl = [t for t in tables if not re.search(rf"\b{re.escape(t)}\b", sql, re.I)]
        miss_cols = {t: [c for c in cols if not re.search(rf"\b{re.escape(c)}\b", sql, re.I)] for t, cols in col_map.items()}
        miss_cols = {t: cs for t, cs in miss_cols.items() if cs}
        alias_note = _alias_mismatch_notes(sql)
        if miss_tbl or miss_cols or alias_note:
            msg = f"Query {idx}: "
            if miss_tbl:
                msg += f"missing tables {miss_tbl}. "
            if miss_cols:
                msg += f"missing columns {miss_cols}. "
            if alias_note:
                msg += alias_note
            reasons.append(msg)
    return "\n".join(reasons) or "Validation failed for unknown reasons."


def _schema_cols_map(filtered_schema: List[Dict[str, Any]]) -> Dict[str, List[str]]:
    return {t["table_name"]: list(t.get("column_info", {}).keys()) for t in filtered_schema}


def _parse_join_condition(cond: str) -> str:
    # Remove cardinality hints like "(1:N)"
    return re.sub(r"\s*\([^)]*\)\s*$", "", cond).strip()


def _build_joins_graph(filtered_schema: List[Dict[str, Any]]) -> Dict[Tuple[str, str], str]:
    edges: Dict[Tuple[str, str], str] = {}
    for t in filtered_schema:
        tname = t.get("table_name")
        joins = t.get("joins", {}) or {}
        for other, cond in joins.items():
            edges[(tname, other)] = _parse_join_condition(cond)
    return edges


def _fallback_sql(tables: List[str], col_map: Dict[str, List[str]], filtered_schema: List[Dict[str, Any]]) -> str:
    if not tables:
        raise RuntimeError("No tables provided for fallback SQL.")

    schema_cols = _schema_cols_map(filtered_schema)
    # Keep only columns that actually exist in schema
    safe_col_map: Dict[str, List[str]] = {}
    for t, cols in col_map.items():
        actual = set(schema_cols.get(t, []))
        wanted = [c for c in cols if c in actual]
        if wanted:
            safe_col_map[t] = wanted

    # If after filtering nothing remains, select a primary key-like column if available
    if not safe_col_map:
        for t, cols in schema_cols.items():
            if tables and t in tables and cols:
                safe_col_map[t] = cols[:1]

    # Build SELECT list
    select_parts: List[str] = []
    for t, cols in safe_col_map.items():
        for c in cols:
            alias = f"{t.replace('.', '_')}_{c}"
            select_parts.append(f"{t}.{c} AS {alias}")
    if not select_parts:
        # As last resort, select ROTN if present
        for t, cols in schema_cols.items():
            if t in tables and "ROTN" in cols:
                select_parts.append(f"{t}.ROTN AS {t.replace('.', '_')}_ROTN")
                break
    select_clause = ",\n       ".join(select_parts) if select_parts else "*"

    # Build JOIN chain from first table
    edges = _build_joins_graph(filtered_schema)
    base = tables[0]
    joined: List[str] = [base]
    joins_sql: List[str] = []
    remaining = [t for t in tables[1:] if t != base]

    while remaining:
        progressed = False
        for t in list(remaining):
            cond = None
            # look for edge between any joined table and t
            for j in joined:
                cond = edges.get((j, t)) or edges.get((t, j))
                if cond:
                    break
            if cond:
                joins_sql.append(f"LEFT JOIN {t} ON {cond}")
                joined.append(t)
                remaining.remove(t)
                progressed = True
        if not progressed:
            # Could not find join path; break and list remaining tables as cross join to avoid failure
            for t in remaining:
                joins_sql.append(f", {t}")
            break

    sql = (
        "SELECT\n       " + select_clause + f"\nFROM {base}\n" +
        ("\n".join(joins_sql) + "\n" if joins_sql else "") +
        "LIMIT 500;"
    )
    print("[SQLGenerator] Using fallback SQL.")
    return sql


# Load the system prompt from static
_PROMPT_PATH = _BASE_DIR / "static" / "sql_generator_system_prompt.txt"
try:
    SYSTEM_PROMPT = _PROMPT_PATH.read_text(encoding="utf-8")
except Exception:
    SYSTEM_PROMPT = "You are a top-tier SQL architect generating Vertica-compatible SELECT queries."


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


def _extract_json_object(text: str) -> str:
    # Remove markdown fences if present
    text = text.strip()
    if text.startswith("```"):
        text = re.sub(r"^```[\w-]*\n", "", text)
        text = re.sub(r"\n```$", "", text)
    start = text.find("{")
    end = text.rfind("}")
    return text[start:end+1] if start != -1 and end != -1 and end > start else text


def _ensure_tables_and_cols(tables: List[str], col_map: Dict[str, List[str]], filtered_schema: List[Dict[str, Any]]) -> Tuple[List[str], Dict[str, List[str]]]:
    """If no tables were inferred from KPIs, derive a safe set from the filtered schema
    and pick at least one column per table (prefer ROTN when present)."""
    if tables:
        return tables, col_map
    schema_cols = _schema_cols_map(filtered_schema)
    derived_tables = list(schema_cols.keys())
    derived_map: Dict[str, List[str]] = {}
    for t, cols in schema_cols.items():
        if not cols:
            continue
        chosen: List[str] = []
        if "ROTN" in cols:
            chosen.append("ROTN")
        # add one more column if available for select variety
        chosen += [c for c in cols if c != "ROTN"][:1]
        derived_map[t] = chosen or cols[:1]
    if derived_tables:
        print("[SQLGenerator] No tables inferred from KPIs; using schema tables:", derived_tables)
    return derived_tables, derived_map


def _generate_sql_string(tables: List[str], col_map: Dict[str, List[str]], kpi_list: List[Dict[str, Any]], filtered_schema: List[Dict[str, Any]]) -> Union[str, List[str]]:
    # Ensure we always have some tables/columns to work with
    tables, col_map = _ensure_tables_and_cols(tables, col_map, filtered_schema)

    schema_str = json.dumps(filtered_schema, indent=2)
    system_prompt_with_schema = SYSTEM_PROMPT.replace("{{table_schema_block}}", schema_str)
    user_msg = (
        f"Tables: {tables}\n"
        f"all_required_columns_by_table: {json.dumps(col_map, ensure_ascii=False)}\n"
        f"KPI list (JSON): {json.dumps(kpi_list, ensure_ascii=False)}\n\n"
        "Return ONLY a JSON object with key 'sql' whose value is either a string SELECT or a list of SELECT strings."
    )

    messages = [
        {"role": "system", "content": system_prompt_with_schema},
        {"role": "user", "content": user_msg},
    ]

    for attempt in range(MAX_RETRIES + 1):
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
        try:
            content = (resp.choices[0].message.content or "").strip()
            print("[SQLGenerator] Raw LLM content:", content[:500])
            json_payload = _extract_json_object(content)
            sql_result = json.loads(json_payload)["sql"]
            if isinstance(sql_result, str):
                sql_result = [sql_result]
            # Print and persist the parsed SQL for debugging
            try:
                print("[SQLGenerator] Parsed sql_result:", sql_result)
                debug_dir = _BASE_DIR / DEBUG_DIR
                debug_dir.mkdir(parents=True, exist_ok=True)
                with open(debug_dir / "sql.pkl", "wb") as fp:
                    pickle.dump(sql_result, fp)
            except Exception as dbg_ex:
                print("[SQLGenerator] Failed to persist sql.pkl:", dbg_ex)
        except Exception:
            sql_result = []

        # Validate; also compute alias mismatch notes to guide retries
        all_valid = True
        alias_notes: List[str] = []
        for q in sql_result:
            if not _valid(q, tables, col_map):
                all_valid = False
            note = _alias_mismatch_notes(q)
            if note:
                alias_notes.append(note)
        if all_valid and not alias_notes:
            return sql_result if len(sql_result) > 1 else sql_result[0]

        if attempt < MAX_RETRIES:
            # Provide targeted feedback with what is missing
            details = _missing_reasons(sql_result, tables, col_map)
            if alias_notes:
                details = (details + "\n" if details else "") + "Alias issues: " + "; ".join(set(alias_notes))
            messages.append(
                {
                    "role": "assistant",
                    "content": (
                        "❌ SQL invalid. Fix and return ONLY JSON {\"sql\": <string or list>}. "
                        f"Tables required: {tables}. Columns required per table: {col_map}.\n"
                        f"Problems detected: {details}"
                    ),
                }
            )

    # As a last resort, synthesize a deterministic SQL to keep pipeline moving
    try:
        # Ensure tables/cols are non-empty before fallback
        tables, col_map = _ensure_tables_and_cols(tables, col_map, filtered_schema)
        return _fallback_sql(tables, col_map, filtered_schema)
    except Exception as ex:
        print("[SQLGenerator] Fallback failed:", ex)

    raise RuntimeError("LLM could not produce a valid SQL query.")


def generate_sql(kpi_list: List[Dict[str, Any]], filtered_schema: List[Dict[str, Any]]) -> Union[str, List[str]]:
    tables, col_map = extract_required_tables_and_columns(kpi_list)
    # Guard against empty extraction early
    tables, col_map = _ensure_tables_and_cols(tables, col_map, filtered_schema)
    return _generate_sql_string(tables, col_map, kpi_list, filtered_schema)
