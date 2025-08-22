from __future__ import annotations

import json
import textwrap
import re
from pathlib import Path
from typing import List, Dict, Any, Optional

import pandas as pd
from openai import OpenAI
from config import OPENAI_API_KEY, LLM_MODEL, OPENAI_BASE_URL

DASHBOARD_FILE: str = "./created_dashboard.py"
DASHBOARD_SYSTEM_PROMPT_PATH = "./static/dashboard_creator_system_prompt.txt"

client = OpenAI(api_key=OPENAI_API_KEY, base_url=OPENAI_BASE_URL or None)

with open(DASHBOARD_SYSTEM_PROMPT_PATH, "r") as f:
    SYSTEM_PROMPT: str = f.read()

def generate_dashboard_code(
    kpi_list: List[Dict[str, Any]],
    creation_sql: str,
    filtered_schema: List[Dict[str, Any]],
    model: str = LLM_MODEL,
    temperature: float = 0.2,
    max_tokens: int = 2000,
) -> str:
    schema_str = json.dumps(filtered_schema, indent=2)
    prompt = SYSTEM_PROMPT.replace("{{creation_sql}}", creation_sql.strip()).replace("{{table_schema_block}}", f"<table_schema>\n{schema_str}\n</table_schema>")

    messages = [
        {"role": "system", "content": prompt},
        {
            "role": "user",
            "content": (
                f"<kpis>\n{kpi_list}\n</kpis>\n\n"
                f"<creation_sql>\n{creation_sql}\n</creation_sql>"
            ),
        },
    ]

    resp = client.chat.completions.create(
        model=model,
        messages=messages,
        temperature=temperature,
        max_tokens=max_tokens,
    )

    raw = resp.choices[0].message.content.strip()

    # Sanitize: ensure with_joined_cte is a function, remove accidental decorators and self-imports
    def _ensure_with_joined_cte(code: str, cte_sql: str) -> str:
        code = code.strip()
        # Remove accidental self-imports like "from created_dashboard import *"
        code = re.sub(r"^\s*from\s+created_dashboard\s+import\s+\*\s*$", "", code, flags=re.M)
        # Remove any accidental decorator usages like @with_joined_cte(...)
        code = re.sub(r"^\s*@with_joined_cte\([^\)]*\)\s*\n", "", code, flags=re.M)
        if "def with_joined_cte(" not in code:
            helper = (
                "\n\n"
                "def with_joined_cte(inner_query: str) -> str:\n"
                "    cte = \"\"\"\n"
                "WITH joined_data AS (\n" + cte_sql.strip() + "\n)\n"
                "\"\"\"\n"
                "    return cte + \"\\n\" + inner_query\n"
            )
            # Insert helper after imports if possible
            m = re.search(r"^(?:from\s+\S+\s+import\s+.*|import\s+\S+).*\n", code, flags=re.M)
            if m:
                insert_at = m.end()
                code = code[:insert_at] + helper + code[insert_at:]
            else:
                code = helper + "\n" + code
        return code

    return _ensure_with_joined_cte(raw, creation_sql)

def write_dashboard_file(code_str: str, path: str | Path = DASHBOARD_FILE) -> Path:
    path = Path(path)
    path.write_text(code_str, encoding="utf-8")
    print(f"✅ Dashboard script written to {path.resolve()}")
    return path
