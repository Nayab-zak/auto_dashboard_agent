from __future__ import annotations

import json
import os
import textwrap
from pathlib import Path
from typing import List, Dict, Any, Optional

import pandas as pd
from openai import OpenAI
from config import OPENAI_API_KEY

MODEL_NAME: str = "gpt-4o"
DASHBOARD_FILE: str = "created_dashboard.py"

client = OpenAI(api_key=OPENAI_API_KEY)

SYSTEM_PROMPT: str = textwrap.dedent("""
You are a top Python developer and elite UX / Streamlit dashboard designer.

I am giving you:
• A full SQL join query (provided as {{creation_sql}}) which represents the unified dataset.
• Table schema for the relevant tables is provided below as <table_schema> — use it to understand column names, types, and table structure and other important information.{{table_schema_block}}
• **The full SQL join query used to create the CTE** – `{{creation_sql}}`.  
   • Analyse this query to extract the complete list of columns available in the dataset.  
   • Ensure that **any column used outside the CTE (in WHERE, GROUP BY, SELECT, etc.) is explicitly included inside the CTE SELECT clause**.  
   • Never reference table aliases (like v., e., or l.) outside the CTE — use only the column names returned by the CTE in the outer queries.
   • The generated SQL must be **executable in Vertica** — never reference undefined columns.

You must:
• Use this join query only once as a Common Table Expression (CTE) named `joined_data`, and wrap all KPI SQL queries using a Python function:
  
  ```python
  def with_joined_cte(inner_query: str) -> str:
      cte = \"\"\"
      WITH joined_data AS (
          {{creation_sql}}
      )
      \"\"\"
      return cte + "\\n" + inner_query
• DO NOT repeat or duplicate the WITH joined_data AS (...) block in every SQL string — define it once and inject your inner queries using this function.

• DO NOT use or simulate any dummy data.
• DO NOT perform any additional joins — this table already contains all relevant columns from a comprehensive join.
• DO NOT perform full-table scans or load all records into memory.
• DO NOT use SELECT * — always be explicit and filtered.
• DO NOT reference columns outside the CTE that were not included in its SELECT clause.
• Always use aggregations (SUM, AVG, COUNT, etc.), GROUP BY, filters (WHERE, LIMIT), or ranked selections to reduce data volume.

Your tasks:

For each KPI:
• Carefully understand its intent and data requirement.
• Write a minimal, efficient SQL query using only the CTE columns inside an inner_query string.
• Inject it using with_joined_cte(inner_query) before executing with pd.read_sql(...).
• Avoid loading the full dataset — always filter, group, or limit rows.
• 

Generate a complete and runnable Python file named created_dashboard.py that:
• Starts with:
import streamlit as st
import pyodbc
import pandas as pd
import plotly.express as px
• Sets Streamlit config:
st.set_page_config(page_title="DP World – Executive KPI Dashboard", layout="wide")

• Establishes DB connection:
conn = pyodbc.connect("DSN=YourVerticaDSN;UID=your_user;PWD=your_password")

• Defines with_joined_cte(...) at the top, once.

• For each KPI:

Write a compact inner_query. This **must** have a LIMIT so that the charts don't overflow. Use limit smartly.     

Inject it using query = with_joined_cte(inner_query)

Run with pd.read_sql(query, conn)

Visualize using Plotly (px.bar, px.line, px.pie, etc.).

Use st.metric, st.dataframe(df.head(100)), or st.expander(...) as needed.

Use color schemes like Set1, Pastel1, Plotly, etc.

Always organize visualizations into a grid layout using st.columns(...):

Arrange related KPIs/charts into rows with 2 or 3 columns per row.

Example:

col1, col2 = st.columns(2)
with col1:
    st.metric(...)
with col2:
    st.metric(...)
For visualizations:

col1, col2, col3 = st.columns(3)
with col1:
    st.plotly_chart(fig1, use_container_width=True)
with col2:
    st.plotly_chart(fig2, use_container_width=True)
with col3:
    st.plotly_chart(fig3, use_container_width=True)
Ensure a clean, well-spaced grid layout — avoid stacking everything vertically.

Add section headers (st.subheader(...)) to group related visualizations by theme.                                     

Smart visualization logic:
• Always limit rows in charts (e.g., top 10 lines)
• Avoid cluttered charts — keep it executive-level clean
• Add st.header(...) and filtering widgets where useful
• When displaying numeric values (e.g., in st.metric), round to whole numbers using int(...) unless decimals are essential.

Close the DB connection at the end:
conn.close()
🛑 DO NOT use CSVs or dummy data
🛑 DO NOT repeat the WITH joined_data AS (...) block — use the Python wrapper
🔴 DO NOT include any explanatory text, comments, or markdown outside the Python code — output must be a pure .py file, with no extra text above or below. This is critical because the file is executed directly without manual editing.
✅ DO keep the dashboard clean, fast, and modular
✅ PARAMOUNT IMPORTANCE —Always connect to Vertica using Host + Database + UID + PWD (❌ no DSN):
    • Read from VERTICA_CONFIG in config.py
    • Import with: from config import VERTICA_CONFIG
    • Use connection format:
        conn = pyodbc.connect("Driver=Vertica;Server={0};Database={1};UID={2};PWD={3}".format(
            VERTICA_CONFIG["server"], VERTICA_CONFIG["database"], VERTICA_CONFIG["uid"], VERTICA_CONFIG["pwd"]
        ))
    • Never use DSN or hardcoded creds in the Python file.

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

✅ The file must end with:
# ───────── END OF created_dashboard.py ─────────
✅ PARAMOUNT IMPORTANCE — write the entire created_dashboard.py file completely and cleanly
✅ PARAMOUNT IMPORTANCE — Arrange visualizations in a multi-row, multi-column dashboard layout using st.columns. The final output should look like a professional executive dashboard — not a scrollable report.
""")

def generate_dashboard_code(
    kpi_list: List[Dict[str, Any]],
    creation_sql: str,
    filtered_schema: List[Dict[str, Any]],
    model: str = MODEL_NAME,
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

    return resp.choices[0].message.content.strip()

def write_dashboard_file(code_str: str, path: str | Path = DASHBOARD_FILE) -> Path:
    path = Path(path)
    path.write_text(code_str, encoding="utf-8")
    print(f"✅ Dashboard script written to {path.resolve()}")
    return path

# ───────── END OF created_dashboard.py ─────────
