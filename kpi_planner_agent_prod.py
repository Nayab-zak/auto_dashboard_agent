from __future__ import annotations

import json
import os
import time
import random
from pathlib import Path
from typing import Any, Dict, List

from openai import OpenAI
from openai.types.chat import ChatCompletion
from openai.types.chat.chat_completion_message_tool_call import ChatCompletionMessageToolCall

from config import (
    OPENAI_API_KEY, OPENAI_BASE_URL, OPENAI_TIMEOUT, LLM_MODEL, KPI_TEMPERATURE,
    LLM_TOP_P, LLM_PRESENCE_PENALTY, LLM_FREQUENCY_PENALTY, LLM_SAMPLING_TOP_K, LLM_REPETITION_PENALTY,
    STRICT_VALIDATION, DEBUG_DIR
)

ALLOWED_CHARTS: List[str] = [
    "bar", "line", "area", "pie", "table", "metric", "gauge", "heatmap", "scatter", "histogram"
]

# Use absolute paths relative to this file
_BASE_DIR = Path(__file__).resolve().parent
SCHEMA_JSON_PATH = str(_BASE_DIR / "agent1_schema_metadata_with_samples.json")

# Initialize client
client = OpenAI(api_key=OPENAI_API_KEY, base_url=OPENAI_BASE_URL, timeout=OPENAI_TIMEOUT)

# Load system prompt from static file
_PROMPT_PATH = _BASE_DIR / "static" / "kpi_planner_system_prompt.txt"
try:
    SYSTEM_PROMPT = _PROMPT_PATH.read_text(encoding="utf-8")
except Exception:
    SYSTEM_PROMPT = (
        "You are a senior analytics consultant and statistician specializing in executive dashboards."
    )


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


def _normalize_name(name: str) -> str:
    import re
    return re.sub(r"\W+", " ", (name or "").lower()).strip()


def _deduplicate_kpis(kpis: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    seen = set()
    out: List[Dict[str, Any]] = []
    for k in kpis:
        norm = _normalize_name(k.get("kpi_name", ""))
        if norm and norm not in seen:
            out.append(k)
            seen.add(norm)
    return out


def _validate(kpis: List[Dict[str, Any]], schema_dict: Dict[str, List[str]]) -> bool:
    if not isinstance(kpis, list):
        return False
    kpis = _deduplicate_kpis(kpis)
    if not (3 <= len(kpis) <= 12):
        return False

    normalized_schema = {
        table.lower(): [col.lower() for col in cols]
        for table, cols in schema_dict.items()
    }

    # Enforce unique KPI names
    names = [_normalize_name(k.get("kpi_name", "")) for k in kpis]
    if len(names) != len(set(names)):
        return False

    for kpi in kpis:
        if kpi.get("chart_type") not in [
            "bar", "line", "area", "pie", "table", "metric", "gauge", "heatmap", "scatter", "histogram"
        ]:
            return False

        required_columns = kpi.get("required_columns", {})
        if not isinstance(required_columns, dict):
            return False
        # Allow subset: at least one required column per referenced table must exist
        for tbl, cols in required_columns.items():
            tbl_l = str(tbl).lower()
            if tbl_l not in normalized_schema:
                return False
            if cols:
                if not any(str(col).lower() in normalized_schema[tbl_l] for col in cols):
                    return False
            elif STRICT_VALIDATION:
                # In strict mode, require at least one column per referenced table
                return False

        # Ensure table_names references are valid
        for tbl in kpi.get("table_names", []):
            if str(tbl).lower() not in normalized_schema:
                return False
            if STRICT_VALIDATION:
                # In strict mode, ensure a required column is declared for each table in table_names
                if not any(str(tbl).lower() == str(t).lower() for t in required_columns.keys()):
                    return False

    return True


def _heuristic_kpis(question: str, schema_dict: Dict[str, List[str]], n: int = 4) -> List[Dict[str, Any]]:
    # Build a minimal valid KPI set using available columns
    tables = list(schema_dict.keys())[:2] or list(schema_dict.keys())
    out: List[Dict[str, Any]] = []
    for i, t in enumerate(tables[:2]):
        cols = schema_dict[t][:3]
        out.append({
            "kpi_name": f"Basic summary from {t}",
            "chart_type": "table" if i == 0 else "bar",
            "required_columns": {t: cols},
            "table_names": [t],
            "why": "Fallback KPI constructed to proceed when LLM fails"
        })
    # cross-table KPI if 2+ tables
    if len(tables) >= 2:
        a, b = tables[:2]
        out.append({
            "kpi_name": f"Join preview {a} x {b}",
            "chart_type": "table",
            "required_columns": {a: schema_dict[a][:1], b: schema_dict[b][:1]},
            "table_names": [a, b],
            "why": "Preview data across two related tables"
        })
    # ensure 3–12
    return out[: max(3, min(12, len(out)))]


def _validation_errors(kpis: List[Dict[str, Any]], schema_dict: Dict[str, List[str]]) -> List[str]:
    errs: List[str] = []
    if not isinstance(kpis, list):
        return ["kpis is not a list"]
    # Dedup first to reflect actual checks
    kpis = _deduplicate_kpis(kpis)
    if not (3 <= len(kpis) <= 12):
        errs.append(f"kpis count {len(kpis)} is outside 3–12")

    normalized_schema = {t.lower(): {c.lower() for c in cols} for t, cols in schema_dict.items()}

    # Check unique names
    names = [_normalize_name(k.get("kpi_name", "")) for k in kpis]
    if len(names) != len(set(names)):
        errs.append("kpi names contain duplicates or near-duplicates")

    for i, kpi in enumerate(kpis):
        ct = kpi.get("chart_type")
        if ct not in ["bar", "line", "area", "pie", "table", "metric", "gauge", "heatmap", "scatter", "histogram"]:
            errs.append(f"kpi[{i}].chart_type '{ct}' not in allowed list")
        rc = kpi.get("required_columns", {})
        if not isinstance(rc, dict):
            errs.append(f"kpi[{i}].required_columns is not an object")
        else:
            for tbl, cols in rc.items():
                tl = str(tbl).lower()
                if tl not in normalized_schema:
                    errs.append(f"kpi[{i}].required_columns references unknown table '{tbl}'")
                    continue
                if cols:
                    if not any(str(col).lower() in normalized_schema[tl] for col in cols):
                        errs.append(f"kpi[{i}] none of required columns exist in table '{tbl}': {cols}")
                elif STRICT_VALIDATION:
                    errs.append(f"kpi[{i}] requires at least one column for table '{tbl}' in strict mode")
        for tbl in kpi.get("table_names", []):
            if str(tbl).lower() not in normalized_schema:
                errs.append(f"kpi[{i}].table_names references unknown table '{tbl}'")
            elif STRICT_VALIDATION and not any(str(tbl).lower() == str(t).lower() for t in rc.keys()):
                errs.append(f"kpi[{i}] must declare required_columns for table '{tbl}' in strict mode")
    return errs


def plan_kpis(question: str, tables: List[str], max_retries: int = 2) -> List[Dict[str, Any]]:
    schema_dict = _get_all_columns(tables)
    col_block = "\n".join([f"{tbl}: {', '.join(cols)}" for tbl, cols in schema_dict.items()])

    instruct = (
        SYSTEM_PROMPT.strip()
        + "\n\nReturn ONLY a single JSON object with this exact shape: {\"kpis\": [ ... ]}.\n"
        + f"Chart types allowed: {ALLOWED_CHARTS}. Use only provided tables and columns.\n"
        + "Avoid repeating the same KPI names across different runs; tailor KPIs specifically to the given user question.\n"
    )

    # Add a small random salt to discourage caching-like behavior in serving layers
    salt = f"salt:{int(time.time()*1000)}:{random.randint(0, 1_000_000)}"

    base_messages: List[Dict[str, Any]] = (
        [{"role": "system", "content": instruct}] +
        _fewshot_messages() +
        [{
            "role": "user",
            "content": f"{salt}\nUser question: {question}\n\nTables: {', '.join(tables)}\nSchemas:\n{col_block}"
        }]
    )

    messages = base_messages.copy()
    debug_dir = _BASE_DIR / DEBUG_DIR
    debug_dir.mkdir(parents=True, exist_ok=True)
    last_errors: List[str] = []

    for attempt in range(max_retries + 1):
        extra_kwargs = {
            "top_p": LLM_TOP_P,
            "presence_penalty": LLM_PRESENCE_PENALTY,
            "frequency_penalty": LLM_FREQUENCY_PENALTY,
        }
        if LLM_SAMPLING_TOP_K is not None:
            extra_kwargs["top_k"] = LLM_SAMPLING_TOP_K  # for ollama/compatible servers
        if LLM_REPETITION_PENALTY is not None:
            extra_kwargs["repetition_penalty"] = LLM_REPETITION_PENALTY

        resp: ChatCompletion = client.chat.completions.create(
            model=LLM_MODEL,
            temperature=KPI_TEMPERATURE,
            messages=messages,
            **extra_kwargs,
        )

        content = (resp.choices[0].message.content or "").strip()
        print("[KPIPlanner] Raw content:", content[:500])
        # persist raw content per attempt
        try:
            (debug_dir / f"kpi_raw_attempt{attempt}.txt").write_text(content, encoding="utf-8")
        except Exception:
            pass

        try:
            start = content.find("{")
            end = content.rfind("}")
            content_json = content[start:end+1] if start != -1 and end != -1 and end > start else content
            parsed = json.loads(content_json)
            kpis = parsed.get("kpis", [])
            # dedup early for stability
            kpis = _deduplicate_kpis(kpis)
            # persist parsed per attempt
            try:
                (debug_dir / f"kpi_parsed_attempt{attempt}.json").write_text(json.dumps({"kpis": kpis}, ensure_ascii=False, indent=2), encoding="utf-8")
            except Exception:
                pass
        except Exception:
            kpis = []

        if _validate(kpis, schema_dict):
            return kpis

        # collect validation errors for diagnosis
        last_errors = _validation_errors(kpis, schema_dict)

        if attempt < max_retries:
            messages.append({
                "role": "assistant",
                "content": (
                    "❌ Validation failed: Provide 3–12 deduplicated KPIs as JSON {\"kpis\": [...]}. Use only listed tables/columns."
                )
            })

    # Final failure with details
    raise RuntimeError("LLM failed to produce a valid KPI list. " + ("; ".join(last_errors) if last_errors else ""))
