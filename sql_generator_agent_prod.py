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
from config import OPENAI_API_KEY, VERTICA_HOST, VERTICA_PORT, VERTICA_DB, VERTICA_USER, VERTICA_PASSWORD

LLM_MODEL = "gpt-4o"
SCHEMA_JSON_PATH = "agent1_schema_metadata_with_samples.json"
MAX_RETRIES = 2

client = OpenAI(api_key=OPENAI_API_KEY)

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

SYSTEM_PROMPT = """
You are a top-tier SQL architect.

Your job is to generate a single SELECT SQL query (or two if absolutely necessary) to retrieve all the columns required to compute the given KPIs.
Each KPI lists required columns and the tables they belong to.

Rules:
1. Use only the provided tables.
2. Use only join keys and table relationships from the schema below. Do not invent any.
3. Prefer to generate a single query using INNER JOINs or LEFT JOINs to merge all tables based on common columns.
4. Only if it’s impossible to include all required columns in one query, return a list of 2 SELECT queries.
5. Never use INSERT, DELETE, DROP, UPDATE or any non-SELECT statement.
6. Always alias columns if names repeat across tables.

Known valid join paths (use only these):
- DPW_DL.Voyages ⇄ DPW_DL.EDW_VOYAGE_BOX_STAT: DPW_DL.Voyages.ROTN = DPW_DL.EDW_VOYAGE_BOX_STAT.ROTN (1:N)
- DPW_DL.EDW_VOYAGE_BOX_STAT ⇄ DPW_DL.Voyages: DPW_DL.EDW_VOYAGE_BOX_STAT.ROTN = DPW_DL.Voyages.ROTN (Many:1)
- DPW_DL.EDW_VOYAGE_BOX_STAT ⇄ DPW_DL.DM_LINES: DPW_DL.EDW_VOYAGE_BOX_STAT.BOX_LINE_ID = DPW_DL.DM_LINES.LINE_ID (Many:1)
- DPW_DL.DM_LINES ⇄ DPW_DL.EDW_VOYAGE_BOX_STAT: DPW_DL.DM_LINES.LINE_ID = DPW_DL.EDW_VOYAGE_BOX_STAT.BOX_LINE_ID (1:N)

<table_schema>
{{table_schema_block}}
</table_schema>

Return your result using function-calling JSON, under the `generate_sql` tool.
If returning multiple queries, return them as a list.

P.S. Paramount importance - WE ARE USING VERTICA TO RUN THESE QUERIES. 
So make sure these queries should run on vertica

Special Data Handling:
- If any column like `year_mth` is in `YYYYMM` format and used for filtering, grouping, or visualization, always convert it to a valid date using:

  TO_DATE(CAST(year_mth AS VARCHAR(6)) || '01', 'YYYYMMDD')

  This turns values like 202407 into a proper DATE: '2024-07-01'.

- Never use `year_mth` as-is in WHERE or GROUP BY clauses when working with dates — always cast it to a DATE first.

The `TERMINAL_ID` column is an integer but must be treated as a **categorical string** for reporting.
    Always convert it to a string prefixed with `'T'` like this:

    `'T' || TERMINAL_ID` or `CONCAT('T', TERMINAL_ID)`

    For example, TERMINAL_ID 1 should be shown as `'T1'`.

- Do not group or filter numerically on `TERMINAL_ID`. Always use the prefixed string version.
WHENEVER USER QUERY ASKS FOR TERMINAL RELATED INSIGHTS LIKE VESSAL COUNT BY TERMINAL ETC, USE TERMINAL_ID. DON'T CONFUSE IT WITH PORT_CODE.
"""


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
        resp = client.chat.completions.create(
            model=LLM_MODEL,
            temperature=0.0,
            messages=messages,
            tools=[GEN_SQL_TOOL],
            tool_choice={"type": "function", "function": {"name": "generate_sql"}},
        )
        call: ChatCompletionMessageToolCall = resp.choices[0].message.tool_calls[0]
        try:
            sql_result = json.loads(call.function.arguments)["sql"]
            if isinstance(sql_result, str):
                sql_result = [sql_result]
        except Exception:
            sql_result = []

        if all(_valid(q, tables, col_map) for q in sql_result):
            return sql_result if len(sql_result) > 1 else sql_result[0]

        if attempt < MAX_RETRIES:
            messages.append(
                {
                    "role": "assistant",
                    "content": (
                        "❌ SQL invalid – make sure it is SELECT-only, uses correct joins across "
                        f"{tables}, and includes all columns from: {col_map}. Try again."
                    ),
                }
            )

    raise RuntimeError("LLM could not produce a valid SQL query.")

def generate_sql(kpi_list: List[Dict[str, Any]], filtered_schema: List[Dict[str, Any]]) -> Union[str, List[str]]:
    tables, col_map = extract_required_tables_and_columns(kpi_list)
    return _generate_sql_string(tables, col_map, kpi_list, filtered_schema)

# Example usage for Vertica connection string:
# conn = pyodbc.connect(f"Driver=Vertica;Server={VERTICA_HOST};Port={VERTICA_PORT};Database={VERTICA_DB};UID={VERTICA_USER};PWD={VERTICA_PASSWORD}")
