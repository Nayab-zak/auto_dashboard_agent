"""
One-page query-to-dashboard service
 • Works with JS (AJAX) or without
 • No Ngrok, no auto-reloader
"""
import os, subprocess, time, socket, pickle, pandas as pd
from flask import Flask, render_template, request, jsonify
import re, textwrap
# ── your agents ───────────────────────────────────────────────────────────────
from table_selector_agent_prod_1 import recommend_table 
from kpi_planner_agent_prod_1 import plan_kpis
from sql_generator_agent_prod_1 import generate_sql
from dashboard_creator_agent_cte_1 import generate_dashboard_code, write_dashboard_file
from typing import Union
import json
# ──────────────────────────────────────────────────────────────────────────────

DASHBOARD_MODULE = "created_dashboard.py"
STREAMLIT_CMD = [
    "streamlit", "run", DASHBOARD_MODULE,
    "--server.headless", "true",
    "--server.enableCORS", "false",
    "--server.enableXsrfProtection", "false",
]
PORT_TIMEOUT = 15                     # seconds to wait for Streamlit

TEMPLATE_FOLDER = "templates"
STATIC_FOLDER = "static"

app = Flask(__name__, template_folder=TEMPLATE_FOLDER, static_folder=STATIC_FOLDER)
_running = {}                         # url: Popen

# ───────────── helpers ─────────────
def _free_port() -> int:
    with socket.socket() as s:
        s.bind(("", 0))
        return s.getsockname()[1]

def _wait(port: int, timeout: int = PORT_TIMEOUT) -> bool:
    start = time.time()
    while time.time() - start < timeout:
        with socket.socket() as s:
            s.settimeout(1)
            try:
                s.connect(("127.0.0.1", port))
                return True
            except OSError:
                time.sleep(.3)
    return False

def _build_dashboard(q: str) -> Union[str, None]:
    tables = recommend_table(q)
    print("Selected Tables: ", tables)
    if tables is None:
        return None  # ← signal to caller that no table was selected

    with open("agent1_schema_metadata_with_samples.json", "r") as f:
        full_schema = json.load(f)

    # Filter schema only for the recommended tables
    filtered_schema = [table for table in full_schema if table["table_name"] in tables]

    kpis = plan_kpis(q, tables)
    print("KPIS:  ", kpis)
    with open("kpi_list.pkl", "wb") as fp:
        pickle.dump(kpis, fp)

    query = generate_sql( kpis, filtered_schema)
    print('query: ', query)

    # Ensure the dashboard creator receives a single SQL string
    if isinstance(query, list):
        if not query:
            raise RuntimeError("SQL generator returned an empty list")
        query = query[0]

    code = generate_dashboard_code(kpis, query, filtered_schema )
    code = _strip_markdown_fence(code)

    # Safety: remove accidental self-imports and decorator misuse
    code = re.sub(r"^\s*from\s+created_dashboard\s+import\s+\*\s*$", "", code, flags=re.M)
    code = re.sub(r"^\s*@with_joined_cte\([^\)]*\)\s*\n", "", code, flags=re.M)

    # Ensure with_joined_cte helper exists
    if "def with_joined_cte(" not in code:
        helper = (
            "def with_joined_cte(inner_query: str) -> str:\n"
            "    cte = \"\"\"\n"
            "WITH joined_data AS (\n" + query.strip() + "\n)\n"
            "\"\"\"\n"
            "    return cte + \"\\n\" + inner_query\n\n"
        )
        code = helper + code

    path = write_dashboard_file(code)
    return os.path.abspath(path)


def _launch(path: str) -> str:
    port = _free_port()
    cmd  = STREAMLIT_CMD + ["--server.port", str(port)]
    proc = subprocess.Popen(cmd, stdout=subprocess.PIPE,
                                 stderr=subprocess.PIPE, text=True)
    if not _wait(port):
        proc.terminate()
        raise RuntimeError("Streamlit failed to start.")
    url = f"http://127.0.0.1:{port}"
    _running[url] = proc
    return url

def _strip_markdown_fence(text: str) -> str:
    """
    Remove leading & trailing fenced code blocks or extract the first fenced block if present.
    If no fences, but the text appears to start with Python code (e.g., 'import '), return as-is.
    Otherwise, return original text.
    """
    if not isinstance(text, str):
        return text
    t = text.strip()
    # Case 1: string starts with a fence – extract the block
    if t.startswith("```"):
        m = re.match(r"```[\w-]*\s*\n([\s\S]*?)\n?```\s*\Z", t)
        if m:
            return textwrap.dedent(m.group(1)).rstrip() + "\n"
    # Case 2: find the first fenced code block anywhere
    fence = re.search(r"```[\w-]*\s*\n([\s\S]*?)\n?```", t)
    if fence:
        return textwrap.dedent(fence.group(1)).rstrip() + "\n"
    # Case 3: heuristic – starts like Python code
    if t.startswith("import ") or t.startswith("from "):
        return t
    return text

# ───────────── routes ─────────────
@app.route("/")
def index():
    return render_template("index.html")

# AJAX endpoint (JS)
@app.route("/api/generate", methods=["POST"])
def api_generate():
    data  = request.get_json(force=True) or {}
    query = (data.get("query") or "").strip()
    if not query:
        return jsonify({"error": "Query is required."}), 400
    try:
        path = _build_dashboard(query)
        if path is None:
            return jsonify({"error": "Sorry, I couldn’t find a relevant dataset for your query. Please try rephrasing it."}), 400
        url = _launch(path)
        return jsonify({"url": url})
    except Exception as e:
        return jsonify({"error": str(e)}), 500

# Non-JS / fallback (GET or POST form)
@app.route("/generate", methods=["GET", "POST"])
def generate():
    query = (request.values.get("query") or "").strip()
    if not query:
        return render_template("result.html", error="Query is required.")
    try:
        path = _build_dashboard(query)
        if path is None:
            return render_template("result.html", error="Sorry, I couldn’t find a relevant dataset for your query. Please try rephrasing it.")
        url = _launch(path)
        return render_template("result.html", url=url)
    except Exception as e:
        return render_template("result.html", error=str(e))

@app.route("/stop")
def stop():
    url = request.args.get("url", "")
    proc = _running.pop(url, None)
    if proc: proc.terminate()
    return f"Stopped {url}" if proc else "No such dashboard."

if __name__ == "__main__":
    app.run("0.0.0.0", 5000, debug=True, use_reloader=False)
