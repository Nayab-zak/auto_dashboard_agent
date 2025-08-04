"""
One-page query-to-dashboard service
 • Works with JS (AJAX) or without
 • No Ngrok, no auto-reloader
"""
import os, subprocess, time, socket, pickle, pandas as pd
from flask import Flask, render_template, request, jsonify
from flask_cors import CORS
import re, textwrap
# ── your agents ───────────────────────────────────────────────────────────────
from table_selector_agent_prod import recommend_table 
from kpi_planner_agent_prod import plan_kpis
from sql_generator_agent_prod import generate_sql
from dashboard_creator_agent_cte import generate_dashboard_code, write_dashboard_file
from typing import Union
import json
from utils.logger import logger
# ──────────────────────────────────────────────────────────────────────────────

DASHBOARD_MODULE = os.path.join(os.path.dirname(__file__), "created_dashboard.py")
STREAMLIT_CMD = [
    "streamlit", "run", DASHBOARD_MODULE,
    "--server.headless", "true",
    "--server.enableCORS", "false",
    "--server.enableXsrfProtection", "false",
]
PORT_TIMEOUT = 15                     # seconds to wait for Streamlit

from flask_cors import CORS

app = Flask(__name__, template_folder="templates", static_folder="static")
# Enable CORS for all routes
CORS(app)
_running = {}                         # url: Popen

# ───────────── helpers ─────────────
def _free_port() -> int:
    with socket.socket() as s:
        s.bind(("", 0))
        return s.getsockname()[1]

def _wait(port: int, timeout: int = PORT_TIMEOUT) -> bool:
    """
    Wait for a port to become available with improved connection checking.
    Also tries to validate if the service is actually ready by checking HTTP response.
    """
    start = time.time()
    while time.time() - start < timeout:
        # First check if port is open using socket
        with socket.socket() as s:
            s.settimeout(1)
            try:
                s.connect(("127.0.0.1", port))
                logger.debug(f"Port {port} is open, checking if service is ready...")
                
                # Give Streamlit a moment to initialize fully
                time.sleep(2)
                
                # Try a simple HTTP request to see if Streamlit is responding
                try:
                    import urllib.request
                    url = f"http://127.0.0.1:{port}/"
                    with urllib.request.urlopen(url, timeout=2) as response:
                        if response.status == 200:
                            logger.info(f"Service on port {port} is fully ready!")
                            return True
                        else:
                            logger.debug(f"Service on port {port} responded with status {response.status}, waiting...")
                except Exception as http_err:
                    logger.debug(f"HTTP check failed for port {port}: {str(http_err)}, but port is open. Considering ready.")
                    # If HTTP check fails but socket connected, consider it ready anyway
                    return True
                
            except (OSError, ConnectionRefusedError) as e:
                logger.debug(f"Waiting for port {port} to be available: {str(e)}")
                time.sleep(1)
    
    logger.warning(f"Timeout waiting for port {port} after {timeout} seconds")
    return False

def _build_dashboard(q: str) -> Union[str, None]:
    try:
        logger.info(f"Building dashboard for query: {q}")
        tables = recommend_table(q)
        logger.info(f"Selected Tables: {tables}")
        if tables is None:
            return None  # ← signal to caller that no table was selected

        # Use absolute path for schema file
        schema_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "agent1_schema_metadata_with_samples.json")
        logger.info(f"Loading schema from: {schema_path}")
        
        if not os.path.exists(schema_path):
            logger.error(f"Schema file not found at {schema_path}")
            raise FileNotFoundError(f"Schema file not found at {schema_path}")
            
        with open(schema_path, "r") as f:
            full_schema = json.load(f)
        
        # Filter schema only for the recommended tables
        filtered_schema = [table for table in full_schema if table["table_name"] in tables]
        
        kpis = plan_kpis(q, tables)
        logger.info(f"KPIs planned: {kpis}")
        
        # Use absolute path for KPI file
        kpi_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "kpi_list.pkl")
        os.makedirs(os.path.dirname(kpi_path), exist_ok=True)
        
        with open(kpi_path, "wb") as fp:
            pickle.dump(kpis, fp)

        query = generate_sql(kpis, filtered_schema)
        logger.info(f"SQL generated: {query[:100]}...")

        code = generate_dashboard_code(kpis, query, filtered_schema)
        code = _strip_markdown_fence(code)

        dashboard_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), DASHBOARD_MODULE)
        path = write_dashboard_file(code, dashboard_path)
        logger.info(f"Dashboard written to: {path}")
        return os.path.abspath(path)
    except Exception as e:
        logger.exception(f"Error building dashboard: {str(e)}")
        raise


def _launch(path: str) -> str:
    max_retries = 3
    last_proc = None
    last_error = None
    
    # Validate path exists
    if not os.path.exists(path):
        raise FileNotFoundError(f"Dashboard file not found at {path}")
    
    for attempt in range(max_retries):
        port = _free_port()
        cmd = STREAMLIT_CMD + ["--server.port", str(port)]
        logger.info(f"Attempt {attempt+1}/{max_retries}: Launching Streamlit on port {port} with command: {' '.join(cmd)}")
        
        # Create logs directory if it doesn't exist
        logs_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'logs')
        os.makedirs(logs_dir, exist_ok=True)
        
        # Log streamlit output to file for debugging
        log_file = os.path.join(logs_dir, f"streamlit_launch_{port}.log")
        with open(log_file, 'w') as f:
            f.write(f"Launch command: {' '.join(cmd)}\n\n")
        
        try:
            proc = subprocess.Popen(
                cmd, 
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE, 
                text=True,
                # Make the dashboard accessible from other machines on the network
                env=dict(os.environ, STREAMLIT_SERVER_HEADLESS="true")
            )
            last_proc = proc
            
            # Start monitoring stdout/stderr in separate threads
            def log_output(stream, prefix, log_file):
                for line in stream:
                    with open(log_file, 'a') as f:
                        f.write(f"{prefix}: {line}")
                    if "error" in line.lower() or "exception" in line.lower():
                        logger.error(f"Streamlit {prefix}: {line.strip()}")
                    else:
                        logger.debug(f"Streamlit {prefix}: {line.strip()}")
            
            import threading
            stdout_thread = threading.Thread(target=log_output, args=(proc.stdout, "STDOUT", log_file))
            stderr_thread = threading.Thread(target=log_output, args=(proc.stderr, "STDERR", log_file))
            stdout_thread.daemon = True
            stderr_thread.daemon = True
            stdout_thread.start()
            stderr_thread.start()
            
            if _wait(port):
                url = f"http://127.0.0.1:{port}"
                _running[url] = proc
                logger.info(f"Streamlit successfully launched at {url}")
                return url
            else:
                last_error = f"Timed out waiting for Streamlit on port {port}"
                proc.terminate()
                logger.warning(f"Attempt {attempt+1} failed: {last_error}")
        except Exception as e:
            last_error = str(e)
            logger.exception(f"Error launching Streamlit (attempt {attempt+1}): {last_error}")
    
    # If all retries fail
    err_output = ""
    if last_proc:
        try:
            err_output = last_proc.stderr.read() if last_proc.stderr else ""
        except:
            err_output = "Could not read process error output"
    
    error_msg = f"All {max_retries} attempts to launch Streamlit failed. Last error: {last_error}. Process output: {err_output}"
    logger.error(error_msg)
    raise RuntimeError(error_msg)

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
    
@app.route("/health")
def health_check():
    """Health check endpoint for monitoring"""
    return jsonify({
        "status": "ok",
        "app": "dashboard_agent",
        "running_dashboards": len(_running)
    })

# AJAX endpoint (JS)
@app.route("/api/generate", methods=["POST"])
def api_generate():
    data  = request.get_json(force=True) or {}
    query = (data.get("query") or "").strip()
    if not query:
        logger.warning("No query provided to /api/generate")
        return jsonify({"error": "Query is required."}), 400
    try:
        path = _build_dashboard(query)
        if path is None:
            logger.warning("No relevant dataset found for query: %s", query)
            return jsonify({"error": "Sorry, I couldn’t find a relevant dataset for your query. Please try rephrasing it."}), 400
        url = _launch(path)
        logger.info("Dashboard launched at %s for query: %s", url, query)
        return jsonify({"url": url})
    except Exception as e:
        logger.exception("Error in /api/generate: %s", str(e))
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
    try:
        # Make sure needed directories exist
        os.makedirs(os.path.join(os.path.dirname(__file__), 'logs'), exist_ok=True)
        os.makedirs(os.path.join(os.path.dirname(__file__), 'temp', 'data'), exist_ok=True)
        
        # Log app start
        logger.info(f"Starting dashboard agent app on 0.0.0.0:5000")
        
        # Use threaded=True to handle concurrent requests better
        app.run("0.0.0.0", 5000, debug=True, use_reloader=False, threaded=True)
    except Exception as e:
        logger.exception(f"Error starting Flask app: {str(e)}")
        print(f"Error starting Flask app: {str(e)}")  # Also print to console in case logging fails
