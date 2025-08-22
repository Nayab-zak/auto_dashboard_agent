from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Any, Dict, List
import re

from openai import OpenAI
from openai.types.chat import ChatCompletion
from openai.types.chat.chat_completion_message_tool_call import ChatCompletionMessageToolCall
from config import OPENAI_API_KEY, LLM_MODEL, OPENAI_BASE_URL, OPENAI_TIMEOUT

ALLOWED_CHARTS: List[str] = [
    "bar", "line", "area", "pie", "table", "metric", "gauge", "heatmap", "scatter", "histogram"
]

SCHEMA_JSON_PATH = "agent1_schema_metadata_with_samples.json"

client = OpenAI(api_key=OPENAI_API_KEY, base_url=OPENAI_BASE_URL or None)
JSON_MODE = bool(OPENAI_BASE_URL)

# Helpers for robust JSON extraction and normalization
def _strip_markdown_fence(text: str) -> str:
    t = text.strip()
    if t.startswith("```") and t.endswith("```"):
        m = re.match(r"```[\w-]*\s*\n([\s\S]*?)\n?```$", t)
        if m:
            return m.group(1).strip()
    return text

def _extract_json_dict(text: str) -> Dict[str, Any]:
    # 1) strip fences
    t = _strip_markdown_fence(text or "")
    # 2) direct parse
    try:
        return json.loads(t)
    except Exception:
        pass
    # 3) find first {...} block
    if "{" in t and "}" in t:
        start = t.find("{")
        end = t.rfind("}")
        if end > start:
            snippet = t[start:end+1]
            try:
                return json.loads(snippet)
            except Exception:
                pass
    # 4) naive single-quote to double-quote replacement as last resort
    try:
        return json.loads(t.replace("'", '"'))
    except Exception:
        return {}

def _normalize_chart_type(ct: str) -> str:
    if not isinstance(ct, str):
        return ""
    s = ct.strip().lower()
    s = s.replace(" chart", "").replace("-chart", "").replace("chart", "")
    # map common variants
    aliases = {
        "column": "bar",
        "columns": "bar",
        "linechart": "line",
        "area chart": "area",
        "hist": "histogram",
        "gauges": "gauge",
    }
    s = aliases.get(s, s)
    return s

def _prune_kpis(kpis: List[Dict[str, Any]], schema_dict: Dict[str, List[str]]) -> List[Dict[str, Any]]:
    norm_schema = {t.lower(): {c.lower() for c in cols} for t, cols in schema_dict.items()}
    cleaned: List[Dict[str, Any]] = []
    for k in kpis:
        if not isinstance(k, dict):
            continue
        ct = _normalize_chart_type(k.get("chart_type", ""))
        if ct not in ALLOWED_CHARTS:
            continue
        rc = k.get("required_columns", {})
        if not isinstance(rc, dict):
            continue
        new_rc: Dict[str, List[str]] = {}
        new_tables: List[str] = []
        for tbl, cols in rc.items():
            if not isinstance(cols, list):
                continue
            t_l = str(tbl).lower()
            if t_l not in norm_schema:
                continue
            valid_cols = [c for c in cols if isinstance(c, str) and c.lower() in norm_schema[t_l]]
            if valid_cols:
                # keep original table key casing
                new_rc[tbl] = valid_cols
                new_tables.append(tbl)
        if not new_rc:
            continue
        new_k = {
            "kpi_name": k.get("kpi_name", "Unnamed KPI"),
            "chart_type": ct,
            "required_columns": new_rc,
            "table_names": new_tables or k.get("table_names", []),
            "why": k.get("why", ""),
        }
        cleaned.append(new_k)
    return cleaned

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

KPI_PLANNER_SYSTEM_PROMPT_PATH = "static/kpi_planner_system_prompt.txt"
with open(KPI_PLANNER_SYSTEM_PROMPT_PATH, "r") as f:
    SYSTEM_PROMPT = f.read()




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

def _fallback_kpis(tables: List[str], schema_dict: Dict[str, List[str]]) -> List[Dict[str, Any]]:
    cols = {t: set(schema_dict.get(t, [])) for t in tables}
    out: List[Dict[str, Any]] = []

    def has(t: str, c: str) -> bool:
        return c in cols.get(t, set())

    # identify common tables
    t_box = None
    for t in tables:
        if t.upper().endswith("EDW_VOYAGE_BOX_STAT"):
            t_box = t
            break
    t_voy = None
    for t in tables:
        if t.upper().endswith("VOYAGES"):
            t_voy = t
            break

    # 1) TEUs by Terminal (if box stats available)
    if t_box and (has(t_box, "TERMINAL_ID") and has(t_box, "TEUS")):
        need = {t_box: [c for c in ["TERMINAL_ID", "TEUS", "YEAR_MTH"] if has(t_box, c)]}
        out.append({
            "kpi_name": "TEUs by Terminal",
            "chart_type": "bar",
            "required_columns": need,
            "table_names": [t_box],
            "why": "Compare throughput by terminal."
        })

    # 2) TEU Monthly Trend
    if t_box and (has(t_box, "YEAR_MTH") and has(t_box, "TEUS")):
        out.append({
            "kpi_name": "TEU Monthly Trend",
            "chart_type": "line",
            "required_columns": {t_box: ["YEAR_MTH", "TEUS"]},
            "table_names": [t_box],
            "why": "Track TEU trend over time."
        })

    # 3) TEUs by Container Length
    if t_box and (has(t_box, "CONTR_LEN") and has(t_box, "TEUS")):
        out.append({
            "kpi_name": "TEUs by Container Length",
            "chart_type": "pie",
            "required_columns": {t_box: ["CONTR_LEN", "TEUS"]},
            "table_names": [t_box],
            "why": "Mix of 20ft/40ft throughput."
        })

    # 4) Completed Vessel Operations (if voyages available)
    if t_voy and (has(t_voy, "DISCH_INV_CODE") and has(t_voy, "LOAD_INV_CODE")):
        need = {t_voy: [c for c in ["DISCH_INV_CODE", "LOAD_INV_CODE", "PORT_CODE", "BI_CREATETIME"] if has(t_voy, c)]}
        out.append({
            "kpi_name": "Completed Vessel Operations",
            "chart_type": "metric",
            "required_columns": need,
            "table_names": [t_voy],
            "why": "Count vessels that completed discharge and load."
        })

    # 5) TEUs per Voyage (join via ROTN if available)
    if t_box and t_voy and has(t_box, "ROTN") and has(t_box, "TEUS") and has(t_voy, "ROTN"):
        need = {
            t_voy: [c for c in ["ROTN", "BI_CREATETIME"] if has(t_voy, c)],
            t_box: [c for c in ["ROTN", "TEUS", "YEAR_MTH"] if has(t_box, c)],
        }
        out.append({
            "kpi_name": "TEUs per Completed Voyage",
            "chart_type": "table",
            "required_columns": need,
            "table_names": [t_voy, t_box],
            "why": "Voyage-level TEUs overview."
        })

    # Add simple templates to guarantee >= 3 KPIs
    if len(out) < 3:
        # TEUs by Status
        if t_box and has(t_box, "STATUS") and has(t_box, "TEUS"):
            out.append({
                "kpi_name": "TEUs by Container Status",
                "chart_type": "bar",
                "required_columns": {t_box: ["STATUS", "TEUS"]},
                "table_names": [t_box],
                "why": "Utilization split across empty/full."
            })
        # TEUs by Shipping Line ID
        if len(out) < 3 and t_box and has(t_box, "BOX_LINE_ID") and has(t_box, "TEUS"):
            out.append({
                "kpi_name": "TEUs by Shipping Line (ID)",
                "chart_type": "bar",
                "required_columns": {t_box: ["BOX_LINE_ID", "TEUS"]},
                "table_names": [t_box],
                "why": "Which lines drive volume."
            })
        # Voyages by Port
        if len(out) < 3 and t_voy and has(t_voy, "PORT_CODE"):
            out.append({
                "kpi_name": "Voyages by Port",
                "chart_type": "bar",
                "required_columns": {t_voy: ["PORT_CODE"]},
                "table_names": [t_voy],
                "why": "Distribution of voyages across ports."
            })

    # Ensure 3–12 KPIs
    return out[:12]

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
    if JSON_MODE:
        base_messages.append({
            "role": "user",
            "content": (
                "Return ONLY strict JSON with key 'kpis' as an array of objects. "
                "Do not include markdown fences or prose."
            ),
        })

    messages = base_messages.copy()
    for attempt in range(max_retries + 1):
        resp: ChatCompletion = client.chat.completions.create(
            model=LLM_MODEL,
            temperature=(0.0 if JSON_MODE else 0.2),
            messages=messages,
            timeout=OPENAI_TIMEOUT,
            **({} if JSON_MODE else {"tools": [GEN_KPI_TOOL], "tool_choice": "auto"}),
        )
        try:
            choices = getattr(resp, "choices", []) or []
            if not choices:
                print(f"[KPI Planner] Attempt {attempt}: empty choices from LLM")
                raise RuntimeError("Empty LLM response")
            if JSON_MODE:
                content = choices[0].message.content or "{}"
                print(f"[KPI Planner] Attempt {attempt} raw content:\n{content}\n")
                obj = _extract_json_dict(content)
                kpis = obj.get("kpis", [])
            else:
                tool_calls = choices[0].message.tool_calls
                if not tool_calls:
                    raise RuntimeError("Function was not called.")
                call = tool_calls[0]
                print(f"[KPI Planner] Attempt {attempt} tool arguments:\n{call.function.arguments}\n")
                kpis = json.loads(call.function.arguments)["kpis"]
        except Exception:
            kpis = []

        # Normalize chart types in-place
        for k in kpis:
            if isinstance(k, dict) and "chart_type" in k:
                k["chart_type"] = _normalize_chart_type(k.get("chart_type", ""))

        if _validate(kpis, schema_dict):
            return kpis

        # Try auto-correction in JSON mode
        if JSON_MODE and kpis:
            fixed = _prune_kpis(kpis, schema_dict)
            if _validate(fixed, schema_dict):
                return fixed

        if attempt < max_retries:
            # Use a user instruction (not assistant) to avoid the model parroting assistant text
            example_tbl = tables[0] if tables else "DPW_DL.Voyages"
            example_cols = (schema_dict.get(example_tbl) or [])[:2]
            example_obj = {
                "kpis": [
                    {
                        "kpi_name": "Example KPI",
                        "chart_type": "bar",
                        "required_columns": {example_tbl: example_cols},
                        "table_names": [example_tbl],
                        "why": "Example only."
                    }
                ]
            }
            messages.append({
                "role": "user",
                "content": (
                    "Return NOW a single JSON object with key 'kpis' (3-12 items). "
                    f"Allowed chart_type: {ALLOWED_CHARTS}. Use only columns from the schema above.\n"
                    "Do not include markdown fences or any prose.\n"
                    f"Example format (adapt with the correct columns):\n{json.dumps(example_obj, ensure_ascii=False)}"
                )
            })
        else:
            # final debug hint
            print("LLM KPI raw content:", getattr(resp.choices[0].message, 'content', ''))

    # Deterministic fallback in JSON mode
    if JSON_MODE:
        fallback = _fallback_kpis(tables, schema_dict)
        if len(fallback) >= 3:
            print("[KPI Planner] Using deterministic fallback KPIs.")
            return fallback

    raise RuntimeError("LLM failed to produce a valid KPI list.")
