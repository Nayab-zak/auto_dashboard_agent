"""
One-page query-to-dashboard service
 • Works with JS (AJAX) or without
 • No Ngrok, no auto-reloader
"""
import os, subprocess, time, socket, pickle, pandas as pd
from flask import Flask, render_template, request, jsonify
import re, textwrap
# ── your agents ───────────────────────────────────────────────────────────────
from table_selector_agent_prod import recommend_table 
from kpi_planner_agent_prod import plan_kpis
from sql_generator_agent_prod import generate_sql
from dashboard_creator_agent_cte import generate_dashboard_code, write_dashboard_file
from typing import Union, List
import json
from pathlib import Path
from config import DEBUG_DIR
# ──────────────────────────────────────────────────────────────────────────────

DASHBOARD_MODULE = "created_dashboard.py"
STREAMLIT_CMD = [
    "streamlit", "run",  # the file to run will be appended dynamically
    "--server.headless", "true",
    "--server.enableCORS", "false",
    "--server.enableXsrfProtection", "false",
]
PORT_TIMEOUT = 15                     # seconds to wait for Streamlit

app = Flask(__name__, template_folder="templates", static_folder="static")
_running = {}                         # url: Popen

# Absolute base dir of this app.py
BASE_DIR = Path(__file__).resolve().parent
DBG_DIR = BASE_DIR / DEBUG_DIR

# Clean old generated files at startup
def _clean_generated_artifacts():
    try:
        dash_path = BASE_DIR / "created_dashboard.py"
        if dash_path.exists():
            dash_path.unlink()
            print(f"[Startup] Removed old {dash_path}")
    except Exception as e:
        print(f"[Startup] Could not remove created_dashboard.py: {e}")
    try:
        old_kpi = BASE_DIR / "deprecated" / "kpi.pkl"
        if old_kpi.exists():
            old_kpi.unlink()
            print(f"[Startup] Removed old {old_kpi}")
    except Exception as e:
        print(f"[Startup] Could not remove deprecated/kpi.pkl: {e}")
    # Remove previous SQL debug artifacts
    try:
        DBG_DIR.mkdir(parents=True, exist_ok=True)
        for fname in ("sql.pkl", "sql_used.pkl", "sql_all.pkl", "kpi_error.log"):
            f = DBG_DIR / fname
            if f.exists():
                f.unlink()
                print(f"[Startup] Removed old {f}")
    except Exception as e:
        print(f"[Startup] Could not remove SQL/KPI debug files: {e}")

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

    # Load schema from project directory, regardless of CWD
    schema_path = BASE_DIR / "agent1_schema_metadata_with_samples.json"
    with open(schema_path, "r") as f:
        full_schema = json.load(f)

    # Filter schema only for the recommended tables
    filtered_schema = [table for table in full_schema if table["table_name"] in tables]

    # Plan KPIs with defensive logging
    try:
        kpis = plan_kpis(q, tables)
    except Exception as e:
        # Ensure debug directory exists and persist error for diagnosis
        DBG_DIR.mkdir(parents=True, exist_ok=True)
        err_file = DBG_DIR / "kpi_error.log"
        err_file.write_text(f"KPI planning failed: {e}", encoding="utf-8")
        print(f"KPI planning failed. See {err_file}")
        raise

    print("KPIS:  ", kpis)
    # Ensure debug directory exists and save KPIs for debugging/inspection (absolute path)
    DBG_DIR.mkdir(parents=True, exist_ok=True)
    kpi_pkl = DBG_DIR / "kpi.pkl"
    with open(kpi_pkl, "wb") as fp:
        pickle.dump(kpis, fp)
    print(f"Saved KPIs to {kpi_pkl.resolve()}")

    query = generate_sql( kpis, filtered_schema)
    print('query: ', query)

    # Normalize to list of SQL strings for dashboard generation
    sql_list: List[str]
    if isinstance(query, list):
        sql_list = [q for q in query if isinstance(q, str) and q.strip()]
    else:
        sql_list = [query] if isinstance(query, str) and query.strip() else []

    # Persist final SQL artifacts
    try:
        with open(DBG_DIR / "sql_used.pkl", "wb") as fp:
            pickle.dump(sql_list[0] if sql_list else "", fp)
        with open(DBG_DIR / "sql_all.pkl", "wb") as fp:
            pickle.dump(sql_list, fp)
        print(f"Saved final SQL to {(DBG_DIR / 'sql_used.pkl').resolve()} and sql_all.pkl")
    except Exception as e:
        print("[Warn] Failed to persist SQL pickles:", e)

    code = generate_dashboard_code(kpis, sql_list, filtered_schema )
    code = _strip_markdown_fence(code)

    # Write dashboard file into project directory so we know where it is
    path = write_dashboard_file(code, path=BASE_DIR / DASHBOARD_MODULE)
    return str(Path(path).resolve())


def _launch(path: str) -> str:
    port = _free_port()
    # Run the specific file path we just created
    cmd  = ["streamlit", "run", path] + STREAMLIT_CMD[2:] + ["--server.port", str(port)]
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
    Remove leading & trailing ``` fences (with or without language tag).
    Returns the inner code or the original text if no fence found.
    """
    # fast path: no opening fence
    if not text.lstrip().startswith("```"):
        return text

    # regex: ```<lang optional>\n  (capture anything)  ``` (closing)
    m = re.match(r"```[\w]*\s*\n([\s\S]*?)\n?```$", text.strip())
    if m:
        return textwrap.dedent(m.group(1)).rstrip() + "\n"
    return text        # fallback: give back unchanged

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
    _clean_generated_artifacts()
    app.run("0.0.0.0", 5000, debug=True, use_reloader=False)
