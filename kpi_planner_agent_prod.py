from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Any, Dict, List

from openai import OpenAI
from openai.types.chat import ChatCompletion
from openai.types.chat.chat_completion_message_tool_call import ChatCompletionMessageToolCall
from config import OPENAI_API_KEY

LLM_MODEL: str = "gpt-4o"
ALLOWED_CHARTS: List[str] = [
    "bar", "line", "area", "pie", "table", "metric", "gauge", "heatmap", "scatter", "histogram"
]

SCHEMA_JSON_PATH = "agent1_schema_metadata_with_samples.json"

client = OpenAI(api_key=OPENAI_API_KEY)

def _get_columns_from_json(table: str, schema_path: str = SCHEMA_JSON_PATH) -> List[str]:
    with open(schema_path, "r") as f:
        schema_data = json.load(f)
    for tbl in schema_data:
        if tbl["table_name"] == table:
            return list(tbl["column_info"].keys())
    raise ValueError(f"Table '{table}' not found in schema file")

def _get_all_columns(tables: List[str], schema_path: str = SCHEMA_JSON_PATH) -> Dict[str, List[str]]:
    return {table: _get_columns_from_json(table, schema_path) for table in tables}

GEN_KPI_TOOL: Dict[str, Any] = {
    "type": "function",
    "function": {
        "name": "generate_kpis",
        "description": (
            "Return 3–12 KPI ideas (each with chart type, required columns by table, and why) "
            "for answering the user's question using the provided tables and their schemas."
        ),
        "parameters": {
            "type": "object",
            "properties": {
                "kpis": {
                    "type": "array",
                    "items": {
                        "type": "object",
                        "properties": {
                            "kpi_name": {"type": "string"},
                            "chart_type": {"type": "string", "enum": ALLOWED_CHARTS},
                            "required_columns": {
                                "type": "object",
                                "additionalProperties": {
                                    "type": "array",
                                    "items": {"type": "string"}
                                }
                            },
                            "table_names": {
                                "type": "array",
                                "items": {"type": "string"}
                            },
                            "why": {"type": "string"},
                        },
                        "required": [
                            "kpi_name", "chart_type", "required_columns", "table_names", "why"
                        ],
                    },
                }
            },
            "required": ["kpis"],
        },
    },
}

SYSTEM_PROMPT = (
    "You are a senior analytics consultant and statistician specializing in executive dashboards.\n"
    "Your job is to deeply understand a user’s business question and return between **9 and 12 useful KPIs and charts/graphs** that help leadership track performance."
    "We need to eventually create a standard dashboard for the leadership using these KPIs and chart visualizations"
    "Our dashboard's standard template includes the first row being 4-5 KPI metrics. The second and third row will be filled with chart/graph visualizations like line, bar, pie, etc."
    "Based on this dashboard template, create meaningful KPIs/graphs"
    "Its of utmost importance that you provide atleast 9 KPIs/Charts."
    "If you're unable to provide atleast 9 KPIs/Charts based on the user query, use your own intelligence to derive the required number of KPI/charts based on other columns from the given tables."
    "In other words, having atleast 9 KPIs/Charts is paramount and non-compromisable."
    "For each KPI/graph, return a JSON object with **all** of the following:\n"
    "- `kpi_name`: clear and specific name\n"
    "- `chart_type`: one of these: bar, line, area, pie, table, metric, gauge, heatmap, scatter, histogram\n"
    "- `required_columns`: a dictionary with keys as table names and values as a list of required columns\n"
    "- `table_names`: the tables used\n"
    "- `why`: why this KPI matters for leadership decision-making\n\n"

    "⚠️ You must only use columns and table names from the provided schema.\n"
    "⚠️ Every KPI must include the correct and complete `required_columns` field.\n"
    "⚠️ Return a JSON that strictly matches the `generate_kpis` function format.\n\n"
    "⚠️ Must ensure that all the KPIs/graphs be executable using provided tables\n\n"

    "Think like a data scientist advising C-level executives. Recommend:\n"
    "- summaries of performance\n"
    "- comparisons across time, lines, terminals, or voyages\n"
    "- top contributors, trends, and distribution breakdowns\n"
    "- statistical KPIs like averages, ratios, and volumes\n"
)




_FEWSHOT_RAW = [
    (
        "Show number of vessels that completed both discharge and load operations in Jebel Ali.",
        ["DPW_DL.Voyages"],
        [
            {
                "kpi_name": "Completed Vessel Operations",
                "chart_type": "metric",
                "required_columns": {
                    "DPW_DL.Voyages": ["DISCH_INV_CODE", "LOAD_INV_CODE", "PORT_CODE"]
                },
                "table_names": ["DPW_DL.Voyages"],
                "why": "To monitor how many vessels completed all operational activities at Jebel Ali port."
            },
            {
                "kpi_name": "Monthly Trend of Completed Operations",
                "chart_type": "line",
                "required_columns": {
                    "DPW_DL.Voyages": ["DISCH_INV_CODE", "LOAD_INV_CODE", "PORT_CODE", "BI_CREATETIME"]
                },
                "table_names": ["DPW_DL.Voyages"],
                "why": "Shows operational activity over time for planning and capacity alignment."
            },
            {
                "kpi_name": "Top Ports by Completed Vessel Operations",
                "chart_type": "bar",
                "required_columns": {
                    "DPW_DL.Voyages": ["PORT_CODE", "DISCH_INV_CODE", "LOAD_INV_CODE"]
                },
                "table_names": ["DPW_DL.Voyages"],
                "why": "Helps identify which ports are most active in completed vessel operations."
            }
        ]
    ),
    (
        "Compare TEUs handled across different terminals in 2024.",
        ["DPW_DL.EDW_VOYAGE_BOX_STAT"],
        [
            {
                "kpi_name": "TEUs by Terminal",
                "chart_type": "bar",
                "required_columns": {
                    "DPW_DL.EDW_VOYAGE_BOX_STAT": ["TERMINAL_ID", "TEUS", "YEAR_MTH"]
                },
                "table_names": ["DPW_DL.EDW_VOYAGE_BOX_STAT"],
                "why": "Helps assess container traffic volume by terminal for planning resources."
            },
            {
                "kpi_name": "TEU Monthly Trend",
                "chart_type": "line",
                "required_columns": {
                    "DPW_DL.EDW_VOYAGE_BOX_STAT": ["YEAR_MTH", "TEUS"]
                },
                "table_names": ["DPW_DL.EDW_VOYAGE_BOX_STAT"],
                "why": "Tracks growth or decline in throughput over time."
            },
            {
                "kpi_name": "TEUs by Container Length",
                "chart_type": "pie",
                "required_columns": {
                    "DPW_DL.EDW_VOYAGE_BOX_STAT": ["CONTR_LEN", "TEUS"]
                },
                "table_names": ["DPW_DL.EDW_VOYAGE_BOX_STAT"],
                "why": "Provides insight into operational asset distribution."
            }
        ]
    ),
    (
        "Which shipping lines had the most TEUs in 2024?",
        ["DPW_DL.EDW_VOYAGE_BOX_STAT", "DPW_DL.DM_LINES"],
        [
            {
                "kpi_name": "Top Shipping Lines by TEUs",
                "chart_type": "pie",
                "required_columns": {
                    "DPW_DL.EDW_VOYAGE_BOX_STAT": ["BOX_LINE_ID", "TEUS"],
                    "DPW_DL.DM_LINES": ["LINE_ID", "LINE_NAME"]
                },
                "table_names": ["DPW_DL.EDW_VOYAGE_BOX_STAT", "DPW_DL.DM_LINES"],
                "why": "Highlights major contributors in container traffic by line."
            },
            {
                "kpi_name": "Monthly TEUs by Shipping Line",
                "chart_type": "line",
                "required_columns": {
                    "DPW_DL.EDW_VOYAGE_BOX_STAT": ["BOX_LINE_ID", "YEAR_MTH", "TEUS"],
                    "DPW_DL.DM_LINES": ["LINE_ID"]
                },
                "table_names": ["DPW_DL.EDW_VOYAGE_BOX_STAT", "DPW_DL.DM_LINES"],
                "why": "Monitors growth patterns across major lines."
            },
            {
                "kpi_name": "Average TEUs per Shipping Line",
                "chart_type": "bar",
                "required_columns": {
                    "DPW_DL.EDW_VOYAGE_BOX_STAT": ["BOX_LINE_ID", "TEUS"],
                    "DPW_DL.DM_LINES": ["LINE_ID"]
                },
                "table_names": ["DPW_DL.EDW_VOYAGE_BOX_STAT", "DPW_DL.DM_LINES"],
                "why": "Gives insight into line efficiency."
            }
        ]
    ),
    (
        "Breakdown of TEUs handled by container length and status.",
        ["DPW_DL.EDW_VOYAGE_BOX_STAT"],
        [
            {
                "kpi_name": "TEUs by Container Length and Status",
                "chart_type": "heatmap",
                "required_columns": {
                    "DPW_DL.EDW_VOYAGE_BOX_STAT": ["CONTR_LEN", "STATUS", "TEUS"]
                },
                "table_names": ["DPW_DL.EDW_VOYAGE_BOX_STAT"],
                "why": "Operational insight into type and condition of containers."
            },
            {
                "kpi_name": "Distribution of Container Lengths",
                "chart_type": "bar",
                "required_columns": {
                    "DPW_DL.EDW_VOYAGE_BOX_STAT": ["CONTR_LEN"]
                },
                "table_names": ["DPW_DL.EDW_VOYAGE_BOX_STAT"],
                "why": "Useful for asset planning and inventory mix."
            },
            {
                "kpi_name": "Volume of Empty vs Full Containers",
                "chart_type": "pie",
                "required_columns": {
                    "DPW_DL.EDW_VOYAGE_BOX_STAT": ["STATUS", "TEUS"]
                },
                "table_names": ["DPW_DL.EDW_VOYAGE_BOX_STAT"],
                "why": "Tracks utilization of container capacity."
            }
        ]
    ),
    (
        "Show completed voyages and their TEU volumes along with line names.",
        ["DPW_DL.Voyages", "DPW_DL.EDW_VOYAGE_BOX_STAT", "DPW_DL.DM_LINES"],
        [
            {
                "kpi_name": "TEUs per Completed Voyage",
                "chart_type": "table",
                "required_columns": {
                    "DPW_DL.Voyages": ["ROTN", "BI_CREATETIME"],
                    "DPW_DL.EDW_VOYAGE_BOX_STAT": ["ROTN", "TEUS"],
                    "DPW_DL.DM_LINES": ["LINE_ID", "LINE_NAME"]
                },
                "table_names": ["DPW_DL.Voyages", "DPW_DL.EDW_VOYAGE_BOX_STAT", "DPW_DL.DM_LINES"],
                "why": "Tracks volume and line ownership for completed voyages."
            },
            {
                "kpi_name": "Average TEUs per Voyage",
                "chart_type": "metric",
                "required_columns": {
                    "DPW_DL.Voyages": ["ROTN"],
                    "DPW_DL.EDW_VOYAGE_BOX_STAT": ["ROTN", "TEUS"]
                },
                "table_names": ["DPW_DL.Voyages", "DPW_DL.EDW_VOYAGE_BOX_STAT"],
                "why": "High-level snapshot of average voyage volume."
            },
            {
                "kpi_name": "Shipping Line Contribution to Voyage TEUs",
                "chart_type": "bar",
                "required_columns": {
                    "DPW_DL.DM_LINES": ["LINE_ID", "LINE_NAME"],
                    "DPW_DL.Voyages": ["LINE_CODE"],
                    "DPW_DL.EDW_VOYAGE_BOX_STAT": ["ROTN", "TEUS"]
                },
                "table_names": ["DPW_DL.DM_LINES", "DPW_DL.Voyages", "DPW_DL.EDW_VOYAGE_BOX_STAT"],
                "why": "Highlights volume contribution by line across voyages."
            }
        ]
    )

]


def _fewshot_messages() -> List[Dict[str, Any]]:
    msgs: List[Dict[str, Any]] = []
    for q, tbls, kpis in _FEWSHOT_RAW:
        all_cols = _get_all_columns(tbls)
        schema_str = "\n".join([f"{t}: {', '.join(cols)}" for t, cols in all_cols.items()])
        msgs.append({
            "role": "user",
            "content": f"Q: {q}\nTables: {', '.join(tbls)}\nSchemas:\n{schema_str}"
        })
        msgs.append({
            "role": "assistant",
            "content": json.dumps({"kpis": kpis}, ensure_ascii=False)
        })
    return msgs

def _validate(kpis: List[Dict[str, Any]], schema_dict: Dict[str, List[str]]) -> bool:
    if not 3 <= len(kpis) <= 12:
        return False

    # Normalize table and column names for case-insensitive match
    normalized_schema = {
        table.lower(): [col.lower() for col in cols]
        for table, cols in schema_dict.items()
    }

    for kpi in kpis:
        if kpi.get("chart_type") not in [
            "bar", "line", "area", "pie", "table", "metric", "gauge", "heatmap", "scatter", "histogram"
        ]:
            return False

        required_columns = kpi.get("required_columns", {})
        if not isinstance(required_columns, dict):
            return False
        if "required_columns" not in kpi:
            print(f"❌ Missing required_columns in: {kpi['kpi_name']}")
            return False

        for tbl, cols in required_columns.items():
            tbl_l = tbl.lower()
            if tbl_l not in normalized_schema:
                return False
            for col in cols:
                if col.lower() not in normalized_schema[tbl_l]:
                    return False

    return True



def plan_kpis(question: str, tables: List[str], max_retries: int = 2) -> List[Dict[str, Any]]:
    schema_dict = _get_all_columns(tables)
    col_block = "\n".join([f"{tbl}: {', '.join(cols)}" for tbl, cols in schema_dict.items()])

    base_messages: List[Dict[str, Any]] = (
        [{"role": "system", "content": SYSTEM_PROMPT}] +
        _fewshot_messages() +
        [{
            "role": "user",
            "content": f"User question: {question}\n\nTables: {', '.join(tables)}\nSchemas:\n{col_block}"
        }]
    )

    messages = base_messages.copy()
    for attempt in range(max_retries + 1):
        resp: ChatCompletion = client.chat.completions.create(
            model=LLM_MODEL,
            temperature=0.2,
            messages=messages,
            tools=[GEN_KPI_TOOL],
            tool_choice="auto",
        )

        try:
            tool_calls = resp.choices[0].message.tool_calls
            if not tool_calls:
                raise RuntimeError("Function was not called.")
            call = tool_calls[0]
            kpis = json.loads(call.function.arguments)["kpis"]
            print(f"\n✅ LLM output on attempt {attempt}:")
            print(json.dumps(kpis, indent=2))
        except Exception as e:
            kpis = []

        if _validate(kpis, schema_dict):
            return kpis

        # ⚠️ Add this line so GPT learns from validation failure
        if attempt < max_retries:
            messages.append({
                "role": "assistant",
                "content": (
                    "❌ Validation failed: Please return between 3–12 KPIs with correct chart_type, valid table names, "
                    "existing column references (from schema), and the required `required_columns` for each table."
                )
            })


    raise RuntimeError("LLM failed to produce a valid KPI list.")
