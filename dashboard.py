import subprocess
import signal
import socket
import os
import re
import threading
from fastapi import FastAPI
from fastapi.responses import HTMLResponse, JSONResponse
import uvicorn

MODELS = {
    "qwen38_ollama": {
        "name": "Qwen 3.8 27B (Q3_K_M)",
        "description": "Qwen3.8-27B Q3_K_M GGUF served via Ollama (262k context)",
        "port": 11434,
        "cmd": [],
        "cwd": "",
        "env": {},
        "protocol": "http",
        "category": "LLM",
        "icon": "brain",
        "color": "#818cf8",
        "tags": ["text-to-text", "ollama"],
        "supports_offload": False,
        "systemd_service": "ollama",
        "ollama_model": "qwen3.8:27b-q3",
    },
}

# Group ordering for display
CATEGORY_ORDER = ["Image", "Video", "Audio", "LLM", "Tools"]

processes: dict[str, subprocess.Popen] = {}
log_files: dict[str, object] = {}
op_lock = threading.Lock()

app = FastAPI()

# ── Power tracking ────────────────────────────────────────────────
import threading, json as _json_mod, datetime as _dt_mod

POWER_LOG = "/mnt/raid1_nvme/JanusPro7b/logs/power_usage.json"

def _load_power_log() -> dict:
    try:
        return _json_mod.loads(open(POWER_LOG).read())
    except Exception:
        return {}

def _save_power_log(data: dict):
    try:
        open(POWER_LOG, "w").write(_json_mod.dumps(data))
    except Exception:
        pass

def _power_sampler():
    """Sample GPU power every 60s and accumulate watt-hours per day."""
    import time
    while True:
        try:
            out = subprocess.run(
                ["nvidia-smi", "--query-gpu=power.draw", "--format=csv,noheader,nounits"],
                capture_output=True, text=True, timeout=5,
            )
            watts = float(out.stdout.strip())
            today = _dt_mod.date.today().isoformat()
            data = _load_power_log()
            # watt-hours = watts * (1min / 60min)
            data[today] = data.get(today, 0) + watts / 60.0
            _save_power_log(data)
        except Exception:
            pass
        time.sleep(60)

_power_thread = threading.Thread(target=_power_sampler, daemon=True)
_power_thread.start()


def is_port_open(port: int) -> bool:
    try:
        with socket.create_connection(("127.0.0.1", port), timeout=1):
            return True
    except (ConnectionRefusedError, TimeoutError, OSError):
        return False


def get_model_status(model_id: str) -> dict:
    model = MODELS[model_id]
    proc = processes.get(model_id)
    managed = proc and proc.poll() is None
    port_up = is_port_open(model["port"])
    if managed:
        return {"status": "ready" if port_up else "starting", "pid": proc.pid, "managed": True}

    processes.pop(model_id, None)
    ollama_model = model.get("ollama_model")
    if ollama_model:
        import json as _json
        import urllib.request
        try:
            with urllib.request.urlopen(f"http://127.0.0.1:{model['port']}/api/ps", timeout=2) as r:
                loaded = [m.get("name", "") for m in _json.loads(r.read()).get("models", [])]
        except Exception:
            loaded = []
        if any(n == ollama_model or n.startswith(ollama_model + ":") for n in loaded):
            return {"status": "ready", "pid": None, "managed": False}
        return {"status": "stopped", "pid": None, "managed": False}
    systemd_service = model.get("systemd_service")
    if systemd_service:
        result = subprocess.run(
            ["systemctl", "is-active", systemd_service], capture_output=True, text=True
        )
        active_state = result.stdout.strip()
        if active_state in ("active", "activating", "reloading"):
            return {"status": "ready" if port_up else "starting", "pid": None, "managed": False}
        return {"status": "stopped", "pid": None, "managed": False}

    if port_up:
        return {"status": "ready", "pid": None, "managed": False}
    return {"status": "stopped", "pid": None, "managed": False}


@app.get("/api/power")
def api_power():
    """Return historical watt-hour usage per day."""
    data = _load_power_log()
    sorted_data = dict(sorted(data.items()))
    today = _dt_mod.date.today().isoformat()
    today_wh = data.get(today, 0)
    return JSONResponse({"by_day": sorted_data, "today_wh": today_wh})


@app.get("/api/tokens")
def api_tokens(days: int = 30):
    """Query opencode SQLite DB for token usage stats."""
    import sqlite3, json as _json, datetime
    db_path = "/root/.local/share/opencode/opencode.db"
    try:
        conn = sqlite3.connect(f"file:{db_path}?mode=ro", uri=True)
        rows = conn.execute(
            "SELECT time_created, data FROM message WHERE json_extract(data, '$.role') = 'assistant'"
        ).fetchall()
        conn.close()
    except Exception as e:
        return JSONResponse({"error": str(e)}, status_code=500)

    by_day: dict = {}
    by_model: dict = {}
    total_input = total_output = total_tokens = 0

    for time_created, data_str in rows:
        try:
            d = _json.loads(data_str)
        except Exception:
            continue
        tokens = d.get("tokens") or {}
        inp = tokens.get("input", 0) or 0
        out = tokens.get("output", 0) or 0
        tot = tokens.get("total", inp + out) or (inp + out)
        model_id = d.get("modelID", "unknown")
        day = datetime.datetime.fromtimestamp(time_created / 1000).strftime("%Y-%m-%d")

        total_input += inp
        total_output += out
        total_tokens += tot

        by_model.setdefault(model_id, {"input": 0, "output": 0, "total": 0})
        by_model[model_id]["input"] += inp
        by_model[model_id]["output"] += out
        by_model[model_id]["total"] += tot

        by_day.setdefault(day, {"input": 0, "output": 0, "total": 0})
        by_day[day]["input"] += inp
        by_day[day]["output"] += out
        by_day[day]["total"] += tot

    limit = None if days <= 0 else days
    sorted_days = dict(sorted(by_day.items())[-limit:] if limit else sorted(by_day.items()))
    sorted_models = dict(sorted(by_model.items(), key=lambda x: -x[1]["total"])[:10])

    return JSONResponse({
        "total_input": total_input,
        "total_output": total_output,
        "total_tokens": total_tokens,
        "by_day": sorted_days,
        "by_model": sorted_models,
    })


@app.get("/api/gpu")
def api_gpu():
    try:
        out = subprocess.run(
            ["nvidia-smi", "--query-gpu=memory.used,memory.total,utilization.gpu,temperature.gpu,power.draw,power.limit,name",
             "--format=csv,noheader,nounits"],
            capture_output=True, text=True, timeout=5,
        )
        parts = [p.strip() for p in out.stdout.strip().split(",")]
        return JSONResponse({
            "vram_used": int(parts[0]),
            "vram_total": int(parts[1]),
            "gpu_util": int(parts[2]),
            "temp": int(parts[3]),
            "power": float(parts[4]),
            "power_limit": float(parts[5]),
            "name": parts[6],
        })
    except Exception:
        return JSONResponse({"error": "nvidia-smi failed"}, status_code=500)


QUANT_PATTERNS = [
    ("NVFP4", "NVFP4"),
    ("IQ3_KS", "IQ3_KS"),
    ("IQ4_XS", "IQ4_XS"),
    ("Q8_K_P", "Q8_K_P"),
    ("Q8_K_XL", "Q8_K_XL"),
    ("Q8_0", "Q8_0"),
    ("Q6_K", "Q6_K"),
    ("Q5_K_M", "Q5_K_M"),
    ("Q4_K_M", "Q4_K_M"),
    ("Q4_0", "Q4_0"),
    ("BF16", "BF16"),
    ("FP16", "FP16"),
    ("FP8", "FP8"),
]


def get_quant(model: dict) -> str:
    """Best-effort detection of the model's quantization/precision."""
    text = " ".join(
        [
            model.get("name", ""),
            model.get("description", ""),
            " ".join(model.get("cmd", [])),
        ]
    ).upper()
    for pattern, label in QUANT_PATTERNS:
        if pattern in text:
            return label
    cmd_args = " ".join(model.get("cmd", []))
    if "--dtype" in cmd_args:
        if "half" in cmd_args or "float16" in cmd_args or "fp16" in cmd_args:
            return "FP16"
        if "bfloat16" in cmd_args or "bf16" in cmd_args:
            return "BF16"
    if re.search(r"(?m)^--bf16\b", "\n".join(model.get("cmd", []))):
        return "BF16"
    return ""


@app.get("/api/status")
def api_status():
    try:
        token_stats = _load_token_stats()
        by_model = token_stats.get("by_model", {})
    except Exception:
        by_model = {}
    result = {}
    for model_id, model in MODELS.items():
        usage = by_model.get(model_id, {}).get("total", 0)
        result[model_id] = {
            "name": model["name"],
            "description": model["description"],
            "port": model["port"],
            "protocol": model["protocol"],
            "category": model["category"],
            "icon": model["icon"],
            "color": model["color"],
            "tags": model.get("tags", []),
            "path": model.get("path", ""),
            "supports_offload": model.get("supports_offload", False),
            "token_usage": usage,
            "quant": get_quant(model),
            "vram_gb": model.get("vram_gb"),
            **get_model_status(model_id),
        }
    return JSONResponse(result)


@app.post("/api/start/{model_id}")
def api_start(model_id: str, offload: bool = False):
    if model_id not in MODELS:
        return JSONResponse({"error": "Unknown model"}, status_code=404)

    with op_lock:
        status = get_model_status(model_id)
        if status["status"] != "stopped":
            return JSONResponse({"message": "Already running", **status})

        model = MODELS[model_id]

        # If managed by systemd, delegate start to systemctl
        systemd_service = model.get("systemd_service")
        if systemd_service:
            subprocess.run(["systemctl", "start", systemd_service], capture_output=True)
            ollama_model = model.get("ollama_model")
            if ollama_model:
                import urllib.request
                payload = ('{"model": "%s", "prompt": "hi", "stream": false, "keep_alive": 86400, "options": {"num_predict": 1, "num_ctx": 512}}' % ollama_model).encode()
                req = urllib.request.Request("http://127.0.0.1:%d/api/generate" % model["port"], data=payload, headers={"Content-Type": "application/json"})
                try:
                    with urllib.request.urlopen(req, timeout=1800) as r:
                        r.read()
                except Exception as e:
                    return JSONResponse({"message": f"Model load failed: {e}"})
                return JSONResponse({"message": "Model loaded into GPU", **get_model_status(model_id)})
            return JSONResponse({"message": "Starting via systemd"})

        env = {**os.environ, **model["env"]}
        cmd = list(model["cmd"])

        if offload and model.get("supports_offload"):
            # vLLM models: add --cpu-offload-gb flag
            if cmd[0].endswith("vllm"):
                cmd.extend(["--cpu-offload-gb", "24"])
            else:
                # Gradio/other apps: set env var for the app to check
                env["CPU_OFFLOAD"] = "1"

        log_path = f"/mnt/raid1_nvme/JanusPro7b/logs/{model_id}.log"
        os.makedirs(os.path.dirname(log_path), exist_ok=True)

        # Close previous log file handle if any
        if model_id in log_files:
            try:
                log_files[model_id].close()
            except Exception:
                pass

        lf = open(log_path, "w")
        log_files[model_id] = lf

        proc = subprocess.Popen(
            cmd,
            cwd=model["cwd"],
            env=env,
            stdout=lf,
            stderr=subprocess.STDOUT,
            preexec_fn=os.setsid,
        )
        processes[model_id] = proc
        return JSONResponse({"message": "Starting", "pid": proc.pid})


def kill_port(port: int):
    """Kill any process listening on the given port."""
    try:
        result = subprocess.run(
            ["fuser", f"{port}/tcp"], capture_output=True, text=True
        )
        for pid_str in result.stdout.split():
            pid = int(pid_str.strip())
            try:
                os.kill(pid, signal.SIGKILL)
            except (ProcessLookupError, PermissionError):
                pass
    except Exception:
        pass


@app.post("/api/stop/{model_id}")
def api_stop(model_id: str):
    if model_id not in MODELS:
        return JSONResponse({"error": "Unknown model"}, status_code=404)

    with op_lock:
        model = MODELS[model_id]

        # If managed by systemd, stop the service so it doesn't restart
        systemd_service = model.get("systemd_service")
        if systemd_service:
            subprocess.run(["systemctl", "stop", systemd_service], capture_output=True)

        proc = processes.get(model_id)
        if proc and proc.poll() is None:
            try:
                os.killpg(os.getpgid(proc.pid), signal.SIGKILL)
                proc.wait()
            except Exception:
                pass
            processes.pop(model_id, None)
            if model_id in log_files:
                try:
                    log_files[model_id].close()
                except Exception:
                    pass
                log_files.pop(model_id, None)
            return JSONResponse({"message": "Stopped"})

        # Not managed by us — kill whatever is on the port
        processes.pop(model_id, None)
        port = model["port"]
        if is_port_open(port):
            kill_port(port)
            return JSONResponse({"message": "Stopped"})
        return JSONResponse({"message": "Already stopped"})


@app.get("/api/logs/{model_id}")
def api_logs(model_id: str):
    if model_id not in MODELS:
        return JSONResponse({"error": "Unknown model"}, status_code=404)
    log_path = f"/mnt/raid1_nvme/JanusPro7b/logs/{model_id}.log"
    text = ""
    try:
        with open(log_path) as f:
            lines = f.readlines()
            text = "".join(lines[-200:])
    except FileNotFoundError:
        pass
    if not text:
        svc = MODELS[model_id].get("systemd_service")
        if svc:
            try:
                out = subprocess.run(
                    ["journalctl", "-u", svc, "-n", "200", "--no-pager"],
                    capture_output=True, text=True, timeout=10,
                )
                text = out.stdout
            except Exception:
                pass
    return JSONResponse({"logs": text})


@app.get("/", response_class=HTMLResponse)
def index():
    return HTML_PAGE


HTML_PAGE = r"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width, initial-scale=1.0">
<title>GPU Model Dashboard</title>
<style>
  @import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700;800&display=swap');
  * { margin: 0; padding: 0; box-sizing: border-box; }
  body {
    font-family: 'Inter', -apple-system, BlinkMacSystemFont, sans-serif;
    background: #0f1117;
    color: #e4e4e7;
    min-height: 100vh;
  }
  .header {
    text-align: center;
    padding: 40px 20px 12px;
  }
  .header h1 {
    font-size: 2em;
    font-weight: 800;
    letter-spacing: -0.03em;
    background: linear-gradient(135deg, #667eea, #f5576c, #a18cd1, #43e97b);
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
    background-size: 300% 300%;
    animation: gradient 8s ease infinite;
  }
  @keyframes gradient {
    0%, 100% { background-position: 0% 50%; }
    50% { background-position: 100% 50%; }
  }
  .header p { color: #71717a; margin-top: 6px; font-size: 0.95em; }
  .gpu-bar {
    max-width: 1200px;
    margin: 16px auto 24px;
    padding: 0 20px;
  }
  .gpu-info {
    background: #1a1b23;
    border: 1px solid #27272a;
    border-radius: 12px;
    padding: 16px 20px;
    font-size: 0.85em;
  }
  .gpu-top {
    display: flex; align-items: center; gap: 12px; margin-bottom: 12px;
  }
  .gpu-info .chip {
    background: #27272a; color: #a1a1aa;
    padding: 3px 10px; border-radius: 6px;
    font-weight: 600; white-space: nowrap;
  }
  .gpu-info .gpu-name { color: #e4e4e7; font-weight: 600; }
  .gpu-info .running-count { color: #22c55e; font-weight: 600; white-space: nowrap; margin-left: auto; }
  .gpu-stats {
    display: grid;
    grid-template-columns: repeat(4, 1fr);
    gap: 10px;
  }
  .gpu-stat {
    background: #0f1117;
    border: 1px solid #27272a;
    border-radius: 8px;
    padding: 10px 12px;
  }
  .gpu-stat-label {
    font-size: 0.72em; color: #52525b; font-weight: 600;
    text-transform: uppercase; letter-spacing: 0.06em;
    margin-bottom: 6px;
  }
  .gpu-stat-value {
    font-size: 1.3em; font-weight: 700; letter-spacing: -0.02em;
  }
  .gpu-stat-sub {
    font-size: 0.72em; color: #52525b; margin-top: 2px;
  }
  .gpu-bar-fill {
    height: 4px; border-radius: 2px; margin-top: 6px;
    background: #27272a; overflow: hidden;
  }
  .gpu-bar-fill-inner {
    height: 100%; border-radius: 2px;
    transition: width 0.5s ease;
  }

  .analytics-bar {
    max-width: 1200px;
    margin: 0 auto 24px;
    padding: 0 20px;
  }
  .analytics-panel {
    background: #1a1b23;
    border: 1px solid #27272a;
    border-radius: 12px;
    padding: 20px;
  }
  .analytics-top {
    display: flex; align-items: center; gap: 12px; margin-bottom: 16px; flex-wrap: wrap;
  }
  .analytics-top .chip { background: #27272a; color: #a1a1aa; padding: 3px 10px; border-radius: 6px; font-weight: 600; font-size: 0.85em; }
  .analytics-title { color: #e4e4e7; font-weight: 600; font-size: 0.85em; }
  .analytics-stats {
    display: grid;
    grid-template-columns: repeat(5, 1fr);
    gap: 10px;
    margin-bottom: 16px;
  }
  .analytics-stat {
    background: #0f1117; border: 1px solid #27272a; border-radius: 8px; padding: 10px 12px;
  }
  .analytics-stat-label { font-size: 0.72em; color: #52525b; font-weight: 600; text-transform: uppercase; letter-spacing: 0.06em; margin-bottom: 6px; }
  .analytics-stat-value { font-size: 1.15em; font-weight: 700; letter-spacing: -0.02em; }
  .analytics-stat-sub { font-size: 0.72em; color: #52525b; margin-top: 2px; }
  .analytics-charts {
    display: grid; grid-template-columns: 2fr 1fr; gap: 16px; margin-top: 4px;
  }
  .analytics-chart-box {
    background: #0f1117; border: 1px solid #27272a; border-radius: 8px; padding: 12px;
    position: relative; height: 180px;
  }
  .analytics-chart-title { font-size: 0.72em; color: #52525b; font-weight: 600; text-transform: uppercase; letter-spacing: 0.06em; margin-bottom: 10px; }
  .kwh-row {
    display: flex; align-items: center; gap: 8px; margin-left: auto;
  }
  .range-btn {
    background: #0f1117; border: 1px solid #27272a; border-radius: 6px;
    color: #71717a; padding: 3px 10px; font-size: 0.75em; cursor: pointer;
    transition: all 0.15s;
  }
  .range-btn:hover { border-color: #3f3f46; color: #e4e4e7; }
  .range-btn.active { background: #27272a; color: #e4e4e7; border-color: #3f3f46; }
  .kwh-row label { font-size: 0.75em; color: #71717a; white-space: nowrap; }
  .kwh-row input {
    background: #27272a; border: 1px solid #3f3f46; border-radius: 6px;
    color: #e4e4e7; padding: 4px 8px; font-size: 0.8em; width: 80px;
    outline: none;
  }
  .kwh-row input:focus { border-color: #667eea; }
  @media (max-width: 700px) {
    .analytics-stats { grid-template-columns: repeat(2, 1fr); }
    .analytics-charts { grid-template-columns: 1fr; }
  }

  .table-bar {
    max-width: 1200px;
    margin: 24px auto 12px;
    padding: 0 20px;
    display: flex;
    align-items: center;
    gap: 12px;
  }
  .search-input {
    flex: 1;
    max-width: 420px;
    background: #1a1b23;
    border: 1px solid #27272a;
    border-radius: 8px;
    color: #e4e4e7;
    padding: 9px 14px;
    font-size: 0.9em;
    font-family: inherit;
    outline: none;
  }
  .search-input:focus { border-color: #667eea; }
  .search-input::placeholder { color: #52525b; }
  .table-count {
    margin-left: auto;
    color: #52525b;
    font-size: 0.8em;
    font-weight: 600;
    white-space: nowrap;
  }
  .table-wrap {
    max-width: 1200px;
    margin: 0 auto;
    padding: 0 20px 12px;
  }
  .table-card {
    background: #1a1b23;
    border: 1px solid #27272a;
    border-radius: 12px;
    overflow-x: auto;
  }
  #model-table {
    width: 100%;
    table-layout: fixed;
    border-collapse: collapse;
    font-size: 0.85em;
  }
  #model-table thead th:nth-child(1) { width: 170px; }
  #model-table thead th:nth-child(2) { width: 74px; }
  #model-table thead th:nth-child(3) { width: 80px; }
  #model-table thead th:nth-child(4) { width: 60px; }
  #model-table thead th:nth-child(5) { width: 130px; }
  #model-table thead th:nth-child(7) { width: 84px; }
  #model-table thead th:nth-child(8) { width: 76px; }
  #model-table thead th:nth-child(9) { width: 84px; }
  #model-table thead th:nth-child(10) { width: 56px; }
  #model-table thead th:nth-child(11) { width: 100px; }
  #model-table thead th {
    text-align: left;
    padding: 12px 10px;
    font-size: 0.7em;
    font-weight: 700;
    text-transform: uppercase;
    letter-spacing: 0.07em;
    color: #71717a;
    border-bottom: 1px solid #27272a;
    cursor: pointer;
    user-select: none;
    white-space: nowrap;
  }
  #model-table thead th:hover { color: #e4e4e7; }
  #model-table thead th.sorted-asc::after { content: ' \25b2'; color: #667eea; }
  #model-table thead th.sorted-desc::after { content: ' \25bc'; color: #667eea; }
  #model-table tbody td {
    padding: 9px 10px;
    border-bottom: 1px solid #202027;
    vertical-align: middle;
  }
  #model-table tbody tr.model-row:hover td { background: #1f2028; }
  .td-name { font-weight: 600; overflow-wrap: break-word; }
  .row-icon {
    display: inline-flex;
    width: 22px; height: 22px;
    border-radius: 6px;
    align-items: center; justify-content: center;
    font-size: 11px; font-weight: 700;
    margin-right: 8px;
  }
  .name-link { color: #e4e4e7; text-decoration: none; }
  .name-link.active:hover { color: #818cf8; text-decoration: underline; }
  .mono { font-family: 'JetBrains Mono', 'Fira Code', monospace; color: #a1a1aa; font-size: 0.92em; }
  .td-tags { white-space: normal; line-height: 1.7; }
  .td-tags .tag-badge { margin: 1px 3px 1px 0; }
  .cat-badge {
    display: inline-block;
    padding: 2px 8px;
    border-radius: 4px;
    font-size: 0.78em;
    font-weight: 700;
    white-space: nowrap;
    margin-right: 8px;
  }
  .td-desc { overflow-wrap: break-word; }
  .desc-text { color: #71717a; font-size: 0.88em; line-height: 1.35; }
  .dim { color: #3f3f46; }
  .td-status { white-space: normal; }
  .tag-badge {
    display: inline-block;
    background: #1e1e2e; color: #71717a;
    padding: 2px 8px; border-radius: 4px;
    font-size: 0.7em; font-weight: 500;
    border: 1px solid #27272a;
  }
  .status-dot { width: 9px; height: 9px; border-radius: 50%; flex-shrink: 0; margin-right: 6px; vertical-align: middle; }
  .status-dot.stopped { background: #52525b; }
  .status-dot.starting { background: #facc15; animation: pulse 1.5s infinite; }
  .status-dot.ready { background: #22c55e; box-shadow: 0 0 8px #22c55e55; }
  @keyframes pulse { 0%, 100% { opacity: 1; } 50% { opacity: 0.3; } }
  .status-text {
    font-size: 0.78em; color: #a1a1aa; font-weight: 600;
    text-transform: uppercase; letter-spacing: 0.06em;
  }
  .switch {
    position: relative; width: 36px; height: 20px; appearance: none;
    background: #27272a; border-radius: 10px; cursor: pointer; transition: background 0.2s;
    outline: none; border: 1px solid #3f3f46; flex-shrink: 0; margin: 0; vertical-align: middle;
  }
  .switch:checked { background: #22c55e; border-color: #22c55e; }
  .switch::after {
    content: ''; position: absolute; top: 2px; left: 2px;
    width: 14px; height: 14px; border-radius: 50%;
    background: #e4e4e7; transition: transform 0.2s;
  }
  .switch:checked::after { transform: translateX(16px); }
  .switch:disabled { opacity: 0.35; cursor: not-allowed; }
  .star {
    background: none; border: none; padding: 2px 6px;
    font-size: 1.15em; line-height: 1; cursor: pointer;
    color: #3f3f46; transition: color 0.15s, transform 0.1s;
  }
  .star:hover { color: #facc15; transform: scale(1.15); }
  .star.on { color: #facc15; text-shadow: 0 0 10px #facc1566; }
  .log-row td { padding: 0 10px 12px; background: #15161c; border-bottom: 1px solid #202027; }
  .row-logbox {
    background: #0f1117; border: 1px solid #27272a;
    border-radius: 8px; padding: 10px;
    font-family: 'JetBrains Mono', 'Fira Code', monospace;
    font-size: 0.72em; color: #9ca3af;
    max-height: 240px; overflow-y: auto;
    white-space: pre-wrap; word-break: break-all;
  }
  .alllogs {
    max-width: 1200px;
    margin: 4px auto 24px;
    padding: 0 20px;
  }
  .alllogs-panel {
    background: #1a1b23;
    border: 1px solid #27272a;
    border-radius: 12px;
    overflow: hidden;
  }
  .alllogs-head {
    display: flex; align-items: center; gap: 10px;
    padding: 14px 18px; cursor: pointer; user-select: none;
  }
  .alllogs-head:hover { background: #1f2028; }
  .alllogs-title { font-weight: 700; font-size: 0.9em; }
  .alllogs-count { color: #52525b; font-size: 0.78em; }
  .alllogs-chev { margin-left: auto; color: #52525b; font-size: 0.9em; }
  .alllogs-body {
    border-top: 1px solid #27272a;
    max-height: 520px; overflow-y: auto;
    padding: 14px 18px;
  }
  .alllogs-body.collapsed { display: none; }
  .alllog-sec { margin-bottom: 16px; }
  .alllog-sec:last-child { margin-bottom: 4px; }
  .alllog-head { display: flex; align-items: center; gap: 8px; margin-bottom: 6px; font-size: 0.82em; }
  .alllog-name { font-weight: 600; }
  .alllog-port { color: #52525b; font-family: monospace; font-size: 0.9em; }
  .alllog-pre {
    background: #0f1117; border: 1px solid #27272a; border-radius: 8px;
    padding: 10px; margin: 0;
    font-family: 'JetBrains Mono', 'Fira Code', monospace;
    font-size: 0.72em; color: #9ca3af;
    max-height: 220px; overflow: auto;
    white-space: pre-wrap; word-break: break-all;
    user-select: text; -webkit-user-select: text;
  }
  .row-logbox-wrap { display: flex; flex-direction: column; }
  .row-logbox-header {
    display: flex; align-items: center; justify-content: space-between;
    padding: 4px 0; margin-bottom: 4px; font-size: 0.75em;
    color: #71717a; font-weight: 600;
  }
  .row-logbox-header span { white-space: nowrap; }
  .row-logbox { user-select: text; -webkit-user-select: text; }
  .copy-btn {
    background: #27272a; border: 1px solid #3f3f46; border-radius: 6px;
    color: #a1a1aa; padding: 3px 10px; font-size: 0.75em; cursor: pointer;
    font-family: inherit; transition: all 0.15s;
  }
  .copy-btn:hover { background: #3f3f46; color: #e4e4e7; }
  .copy-btn.copied { background: #22c55e; color: #fff; border-color: #22c55e; }
  .footer {
    text-align: center; padding: 24px; color: #3f3f46; font-size: 0.75em;
  }
</style>
</head>
<body>

<div class="header">
  <h1>GPU Model Dashboard</h1>
  <p>NVIDIA RTX PRO 6000 &middot; 96 GB VRAM</p>
</div>

<div class="gpu-bar">
  <div class="gpu-info">
    <div class="gpu-top">
      <span class="chip">GPU</span>
      <span class="gpu-name" id="gpu-name">...</span>
      <span class="running-count" id="running-count">0 running</span>
    </div>
    <div class="gpu-stats">
      <div class="gpu-stat">
        <div class="gpu-stat-label">VRAM</div>
        <div class="gpu-stat-value" id="gpu-vram">--</div>
        <div class="gpu-stat-sub" id="gpu-vram-sub"></div>
        <div class="gpu-bar-fill"><div class="gpu-bar-fill-inner" id="gpu-vram-bar" style="width:0%;background:#667eea"></div></div>
      </div>
      <div class="gpu-stat">
        <div class="gpu-stat-label">Core Usage</div>
        <div class="gpu-stat-value" id="gpu-util">--</div>
        <div class="gpu-stat-sub">&nbsp;</div>
        <div class="gpu-bar-fill"><div class="gpu-bar-fill-inner" id="gpu-util-bar" style="width:0%;background:#22c55e"></div></div>
      </div>
      <div class="gpu-stat">
        <div class="gpu-stat-label">Temperature</div>
        <div class="gpu-stat-value" id="gpu-temp">--</div>
        <div class="gpu-stat-sub" id="gpu-temp-sub">&nbsp;</div>
        <div class="gpu-bar-fill"><div class="gpu-bar-fill-inner" id="gpu-temp-bar" style="width:0%;background:#facc15"></div></div>
      </div>
      <div class="gpu-stat">
        <div class="gpu-stat-label">Power</div>
        <div class="gpu-stat-value" id="gpu-power">--</div>
        <div class="gpu-stat-sub" id="gpu-power-sub"></div>
        <div class="gpu-bar-fill"><div class="gpu-bar-fill-inner" id="gpu-power-bar" style="width:0%;background:#f5576c"></div></div>
      </div>
    </div>
  </div>
</div>

<div class="analytics-bar">
  <div class="analytics-panel">
    <div class="analytics-top">
      <span class="chip">OPENCODE</span>
      <span class="analytics-title">Token Usage &amp; Power Cost</span>
      <div style="display:flex;gap:4px;margin-left:12px">
        <button class="range-btn active" onclick="setRange(7)">7d</button>
        <button class="range-btn" onclick="setRange(30)">30d</button>
        <button class="range-btn" onclick="setRange(90)">90d</button>
        <button class="range-btn" onclick="setRange(365)">1y</button>
        <button class="range-btn" onclick="setRange(0)">All</button>
      </div>
      <div class="kwh-row" style="margin-left:auto">
        <label>$/kWh</label>
        <input type="number" id="kwh-rate" value="0.12" step="0.01" min="0" onchange="updateCost(); updateDailyChart()">
      </div>
    </div>
    <div class="analytics-stats">
      <div class="analytics-stat">
        <div class="analytics-stat-label">Total Tokens</div>
        <div class="analytics-stat-value" id="stat-total">--</div>
        <div class="analytics-stat-sub" id="stat-range-label">last 7 days</div>
      </div>
      <div class="analytics-stat">
        <div class="analytics-stat-label">Input Tokens</div>
        <div class="analytics-stat-value" id="stat-input">--</div>
        <div class="analytics-stat-sub" id="stat-input-label">last 7 days</div>
      </div>
      <div class="analytics-stat">
        <div class="analytics-stat-label">Output Tokens</div>
        <div class="analytics-stat-value" id="stat-output">--</div>
        <div class="analytics-stat-sub" id="stat-output-label">last 7 days</div>
      </div>
      <div class="analytics-stat">
        <div class="analytics-stat-label">Power Cost/hr</div>
        <div class="analytics-stat-value" id="stat-cost-hr">--</div>
        <div class="analytics-stat-sub" id="stat-cost-sub">at current draw</div>
      </div>
      <div class="analytics-stat">
        <div class="analytics-stat-label">Est. Today Cost</div>
        <div class="analytics-stat-value" id="stat-cost-day">--</div>
        <div class="analytics-stat-sub">power only</div>
      </div>
    </div>
    <div class="analytics-charts">
      <div class="analytics-chart-box">
        <div style="display:flex;align-items:center;justify-content:space-between;margin-bottom:10px">
          <div class="analytics-chart-title" style="margin-bottom:0" id="chart-daily-title">Tokens per Day</div>
          <div style="display:flex;gap:3px">
            <button class="range-btn active" id="toggle-tokens" onclick="setChartMode('tokens')">Tokens</button>
            <button class="range-btn" id="toggle-cost" onclick="setChartMode('cost')">$ Cost</button>
          </div>
        </div>
        <canvas id="chart-daily"></canvas>
      </div>
      <div class="analytics-chart-box">
        <div class="analytics-chart-title">Top Models by Tokens</div>
        <canvas id="chart-models"></canvas>
      </div>
    </div>
  </div>
</div>

<div class="table-bar">
  <input type="search" class="search-input" id="search" placeholder="Search models, tags, ports, quant…">
  <span class="table-count" id="table-count"></span>
</div>

<div class="table-wrap">
  <div class="table-card">
    <table id="model-table">
      <thead>
        <tr>
          <th data-key="name">Name</th>
          <th data-key="quant">Quant</th>
          <th data-key="vram">VRAM Usage</th>
          <th data-key="port">Port</th>
          <th data-key="tags">Tags</th>
          <th data-key="type">Type / Description</th>
          <th data-key="offload">CPU Offload</th>
          <th data-key="logs">Show Logs</th>
          <th data-key="run">Start/Stop</th>
          <th data-key="fav">Favorite</th>
          <th data-key="status">Status</th>
        </tr>
      </thead>
      <tbody id="model-tbody"></tbody>
    </table>
  </div>
</div>

<div class="alllogs">
  <div class="alllogs-panel">
    <div class="alllogs-head" onclick="toggleAllLogs()">
      <span class="alllogs-title">All Model Logs</span>
      <span class="alllogs-count" id="alllogs-count">collapsed</span>
      <span class="alllogs-chev" id="alllogs-chev">▸</span>
    </div>
    <div class="alllogs-body collapsed" id="alllogs-body"></div>
  </div>
</div>

<div class="footer">Load &middot; Use &middot; Unload</div>

<script src="https://cdn.jsdelivr.net/npm/chart.js@4.4.0/dist/chart.umd.min.js"></script>
<script>
const HOST = location.hostname;
const tbody = document.getElementById('model-tbody');
let currentModels = {};
let sortKey = 'fav';
let sortDir = -1;
let searchQuery = '';
let rowMap = {};
let logIntervals = {};
let favorites = new Set();
try { favorites = new Set(JSON.parse(localStorage.getItem('fav_models') || '[]')); } catch (e) {}

function saveFavs() {
  try { localStorage.setItem('fav_models', JSON.stringify([...favorites])); } catch (e) {}
}

function toggleFav(id) {
  if (favorites.has(id)) favorites.delete(id); else favorites.add(id);
  saveFavs();
  const star = document.getElementById('star-' + id);
  if (star) star.classList.toggle('on', favorites.has(id));
  if (sortKey === 'fav') renderAll(currentModels);
}

function sortValue(id, m, key) {
  switch (key) {
    case 'name': return m.name.toLowerCase();
    case 'quant': return (m.quant || '').toLowerCase() || '~';
    case 'vram': return m.vram_gb == null ? -1 : m.vram_gb;
    case 'port': return m.port;
    case 'tags': return (m.tags || []).join(', ').toLowerCase();
    case 'type': return (m.category || '') + ' :: ' + m.name.toLowerCase();
    case 'offload': return m.supports_offload ? 1 : 0;
    case 'logs': return rowMap[id] && rowMap[id].logOpen ? 1 : 0;
    case 'run': return m.status === 'stopped' ? 0 : 1;
    case 'fav': return favorites.has(id) ? 1 : 0;
    case 'status': { const o = { ready: 0, starting: 1, stopped: 2 }; return o[m.status] != null ? o[m.status] : 3; }
  }
  return '';
}

function sortRows() {
  const ids = Object.keys(currentModels);
  ids.sort((a, b) => {
    const va = sortValue(a, currentModels[a], sortKey);
    const vb = sortValue(b, currentModels[b], sortKey);
    let r = va < vb ? -1 : (va > vb ? 1 : 0);
    if (r === 0) return currentModels[a].name.localeCompare(currentModels[b].name);
    return r * sortDir;
  });
  return ids;
}

function matchesSearch(m) {
  if (!searchQuery) return true;
  const q = searchQuery.toLowerCase();
  const hay = [m.name, m.description || '', m.category || '', String(m.port), (m.tags || []).join(' '), m.quant || ''].join(' ').toLowerCase();
  return hay.includes(q);
}

document.querySelectorAll('#model-table th[data-key]').forEach(th => {
  th.addEventListener('click', () => {
    const key = th.dataset.key;
    if (sortKey === key) sortDir *= -1; else { sortKey = key; sortDir = 1; }
    document.querySelectorAll('#model-table th[data-key]').forEach(h => h.classList.remove('sorted-asc', 'sorted-desc'));
    th.classList.add(sortDir === 1 ? 'sorted-asc' : 'sorted-desc');
    renderAll(currentModels);
  });
});

document.getElementById('search').addEventListener('input', e => {
  searchQuery = e.target.value.trim();
  renderAll(currentModels);
});

document.querySelector('#model-table th[data-key="fav"]').classList.add('sorted-desc');

function buildRow(id, m) {
  const tr = document.createElement('tr');
  tr.className = 'model-row';
  const initial = (m.name || '?').trim().charAt(0);
  tr.innerHTML = `
    <td class="td-name"><span class="row-icon" style="background:${m.color}18;color:${m.color}">${initial}</span><a class="name-link" id="link-${id}" target="_blank" rel="noopener">${m.name}</a></td>
    <td class="mono">${m.quant || '—'}</td>
    <td class="mono">${m.vram_gb ? '~' + m.vram_gb + ' GB' : '—'}</td>
    <td class="mono">${m.port}</td>
    <td class="td-tags">${(m.tags || []).map(t => `<span class="tag-badge">${t}</span>`).join('')}</td>
    <td class="td-desc"><span class="cat-badge" style="background:${m.color}18;color:${m.color}">${m.category}</span><span class="desc-text">${m.description}</span></td>
    <td>${m.supports_offload ? `<input type="checkbox" class="switch" id="offload-${id}">` : '<span class="dim">—</span>'}</td>
    <td><input type="checkbox" class="switch" onchange="toggleRowLogs('${id}', this.checked)"></td>
    <td><input type="checkbox" class="switch" id="runsw-${id}" onchange="toggleRun('${id}', this.checked)" disabled></td>
    <td><button class="star" id="star-${id}" title="favorite" onclick="toggleFav('${id}')">&#9733;</button></td>
    <td class="td-status"><span class="status-dot" id="dot-${id}"></span><span class="status-text" id="statustext-${id}"></span></td>`;
  const logTr = document.createElement('tr');
  logTr.className = 'log-row';
  logTr.style.display = 'none';
  logTr.innerHTML = `<td colspan="11"><div class="row-logbox" id="rowlog-${id}"></div></td>`;
  rowMap[id] = { tr: tr, logTr: logTr, logOpen: false };
  tbody.appendChild(tr);
  tbody.appendChild(logTr);
}

function updateRow(id, m) {
  const dot = document.getElementById('dot-' + id);
  const st = document.getElementById('statustext-' + id);
  if (dot) dot.className = 'status-dot ' + m.status;
  if (st) st.textContent = m.status === 'ready' ? 'Running' : m.status === 'starting' ? 'Starting…' : 'Stopped';
  const link = document.getElementById('link-' + id);
  if (link) {
    if (m.status === 'ready') {
      link.href = (m.protocol || 'http') + '://' + HOST + ':' + m.port + (m.path || '');
      link.classList.add('active');
    } else {
      link.removeAttribute('href');
      link.classList.remove('active');
    }
  }
  const runsw = document.getElementById('runsw-' + id);
  if (runsw) { runsw.checked = m.status !== 'stopped'; runsw.disabled = m.status === 'starting'; }
  const off = document.getElementById('offload-' + id);
  if (off) off.disabled = m.status !== 'stopped';
  const star = document.getElementById('star-' + id);
  if (star) star.classList.toggle('on', favorites.has(id));
}

function renderAll(models) {
  currentModels = models;
  let runCount = 0;
  for (const [id, m] of Object.entries(models)) {
    if (!rowMap[id]) buildRow(id, m);
    if (m.status !== 'stopped') runCount++;
    updateRow(id, m);
  }
  document.getElementById('running-count').textContent = runCount + ' model' + (runCount !== 1 ? 's' : '') + ' running';

  for (const id of sortRows()) {
    tbody.appendChild(rowMap[id].tr);
    tbody.appendChild(rowMap[id].logTr);
  }

  let visible = 0;
  for (const [id, m] of Object.entries(models)) {
    const show = matchesSearch(m);
    if (show) visible++;
    rowMap[id].tr.style.display = show ? '' : 'none';
    rowMap[id].logTr.style.display = (show && rowMap[id].logOpen) ? '' : 'none';
  }
  document.getElementById('table-count').textContent = visible + ' / ' + Object.keys(models).length + ' models';
  document.getElementById('alllogs-count').textContent = allLogsOpen ? runCount + ' model' + (runCount !== 1 ? 's' : '') + ' streaming' : 'collapsed';
}

async function refresh() {
  try {
    const r = await fetch('/api/status');
    renderAll(await r.json());
  } catch (e) {}
}

async function toggleRun(id, on) {
  if (on) {
    const cb = document.getElementById('offload-' + id);
    const offload = cb && cb.checked ? 'true' : 'false';
    await fetch('/api/start/' + id + '?offload=' + offload, { method: 'POST' });
  } else {
    await fetch('/api/stop/' + id, { method: 'POST' });
  }
  setTimeout(refresh, 600);
}

async function fetchRowLog(id) {
  const e = rowMap[id];
  if (!e || !e.logOpen) return;
  const box = document.getElementById('rowlog-' + id);
  if (!box) return;
  try {
    const wasAtBottom = box.scrollHeight - box.scrollTop - box.clientHeight < 40;
    const r = await fetch('/api/logs/' + id);
    const d = await r.json();
    box.textContent = d.logs || '(no logs yet)';
    if (wasAtBottom) box.scrollTop = box.scrollHeight;
  } catch (err) {
    box.textContent = '(error loading logs)';
  }
}

function toggleRowLogs(id, open) {
  const e = rowMap[id];
  if (!e) return;
  e.logOpen = open;
  e.logTr.style.display = open ? '' : 'none';
  if (open) {
    fetchRowLog(id);
    if (!logIntervals[id]) logIntervals[id] = setInterval(() => fetchRowLog(id), 3000);
  } else if (logIntervals[id]) {
    clearInterval(logIntervals[id]);
    delete logIntervals[id];
  }
}

let allLogsOpen = false;

function toggleAllLogs() {
  allLogsOpen = !allLogsOpen;
  document.getElementById('alllogs-body').classList.toggle('collapsed', !allLogsOpen);
  document.getElementById('alllogs-chev').textContent = allLogsOpen ? '\u25be' : '\u25b8';
  document.getElementById('alllogs-count').textContent = allLogsOpen ? 'streaming…' : 'collapsed';
  if (allLogsOpen) updateAllLogs();
}

function runningIds() {
  return Object.keys(currentModels).filter(id => currentModels[id].status !== 'stopped');
}

async function updateAllLogs() {
  const ids = runningIds();
  if (!allLogsOpen) return;
  const body = document.getElementById('alllogs-body');
  for (const id of ids) {
    const m = currentModels[id];
    let pre = document.getElementById('alllogpre-' + id);
    if (!pre) {
      const sec = document.createElement('div');
      sec.className = 'alllog-sec';
      sec.dataset.id = id;
      sec.innerHTML = `<div class="alllog-head"><span class="status-dot ${m.status}"></span><span class="alllog-name">${m.name}</span><span class="alllog-port">${m.protocol || 'http'}://${HOST}:${m.port}</span></div><pre class="alllog-pre" id="alllogpre-${id}"></pre>`;
      body.appendChild(sec);
      pre = document.getElementById('alllogpre-' + id);
    }
    (async () => {
      try {
        const wasAtBottom = pre.scrollHeight - pre.scrollTop - pre.clientHeight < 40;
        const r = await fetch('/api/logs/' + id);
        const d = await r.json();
        pre.textContent = d.logs || '(no logs yet)';
        if (wasAtBottom) pre.scrollTop = pre.scrollHeight;
      } catch (e) {}
    })();
  }
  for (const sec of Array.from(body.querySelectorAll('.alllog-sec'))) {
    if (!ids.includes(sec.dataset.id)) sec.remove();
  }
}

// ── Token analytics ──────────────────────────────────────────────
let tokenData = null;
let powerByDay = {};
let currentPowerW = 0;
let todayWh = 0;
let chartDaily = null;
let chartModels = null;
let currentRange = 7;
let chartMode = 'tokens';

function fmtNum(n) {
  if (n >= 1e9) return (n/1e9).toFixed(2) + 'B';
  if (n >= 1e6) return (n/1e6).toFixed(1) + 'M';
  if (n >= 1e3) return (n/1e3).toFixed(1) + 'K';
  return n.toString();
}

function setRange(days) {
  currentRange = days;
  document.querySelectorAll('.range-btn').forEach(b => {
    const label = b.textContent.trim();
    const match = days === 7 && label === '7d' || days === 30 && label === '30d' ||
                  days === 90 && label === '90d' || days === 365 && label === '1y' ||
                  days === 0 && label === 'All';
    b.classList.toggle('active', match);
  });
  const rangeLabel = days === 0 ? 'all time' : 'last ' + days + ' days';
  document.getElementById('stat-range-label').textContent = rangeLabel;
  document.getElementById('stat-input-label').textContent = rangeLabel;
  document.getElementById('stat-output-label').textContent = rangeLabel;
  refreshTokens();
}

function updateCost() {
  const rate = parseFloat(document.getElementById('kwh-rate').value) || 0.12;
  if (!currentPowerW) return;
  // Cost/hr from current draw
  const costHr = (currentPowerW / 1000) * rate;
  document.getElementById('stat-cost-hr').textContent = '$' + costHr.toFixed(3);
  document.getElementById('stat-cost-sub').textContent = currentPowerW.toFixed(0) + 'W @ $' + rate + '/kWh';
  // Today cost from actual accumulated watt-hours
  const costToday = (todayWh / 1000) * rate;
  document.getElementById('stat-cost-day').textContent = '$' + costToday.toFixed(3);
}

async function refreshPower() {
  try {
    const r = await fetch('/api/power');
    const d = await r.json();
    todayWh = d.today_wh || 0;
    powerByDay = d.by_day || {};
    updateCost();
  } catch(e) {}
}

function setChartMode(mode) {
  chartMode = mode;
  document.getElementById('toggle-tokens').classList.toggle('active', mode === 'tokens');
  document.getElementById('toggle-cost').classList.toggle('active', mode === 'cost');
  updateDailyChart();
}

function updateDailyChart() {
  if (!chartDaily || !tokenData) return;
  const rate = parseFloat(document.getElementById('kwh-rate').value) || 0.12;
  const days = Object.keys(tokenData.by_day);
  const shortDays = days.map(d => d.slice(5));

  if (chartMode === 'tokens') {
    document.getElementById('chart-daily-title').textContent = 'Tokens per Day';
    chartDaily.data.labels = shortDays;
    chartDaily.data.datasets = [
      { label: 'Input', data: days.map(k => tokenData.by_day[k].input), backgroundColor: 'rgba(102,126,234,0.7)', stack: 's' },
      { label: 'Output', data: days.map(k => tokenData.by_day[k].output), backgroundColor: 'rgba(34,197,94,0.7)', stack: 's' },
    ];
    chartDaily.options.scales.y.ticks.callback = v => fmtNum(v);
  } else {
    document.getElementById('chart-daily-title').textContent = 'Power Cost per Day ($)';
    chartDaily.data.labels = shortDays;
    chartDaily.data.datasets = [
      { label: 'Cost ($)', data: days.map(k => {
          const wh = powerByDay[k] || 0;
          return +((wh / 1000) * rate).toFixed(4);
        }),
        backgroundColor: 'rgba(249,115,22,0.7)', stack: 's'
      },
    ];
    chartDaily.options.scales.y.ticks.callback = v => '$' + v.toFixed(3);
  }
  chartDaily.update();
}

async function refreshTokens() {
  try {
    const url = '/api/tokens?days=' + currentRange;
    const r = await fetch(url);
    const d = await r.json();
    if (d.error) return;
    tokenData = d;
    // Sum totals from the filtered day range (not all-time)
    let rangeInput = 0, rangeOutput = 0, rangeTotal = 0;
    for (const v of Object.values(d.by_day)) {
      rangeInput += v.input;
      rangeOutput += v.output;
      rangeTotal += v.total;
    }
    document.getElementById('stat-total').textContent = fmtNum(rangeTotal);
    document.getElementById('stat-input').textContent = fmtNum(rangeInput);
    document.getElementById('stat-output').textContent = fmtNum(rangeOutput);

    // Daily chart
    const days = Object.keys(d.by_day);
    const shortDays = days.map(k => k.slice(5));
    if (!chartDaily) {
      const ctx = document.getElementById('chart-daily').getContext('2d');
      chartDaily = new Chart(ctx, {
        type: 'bar',
        data: { labels: shortDays, datasets: [] },
        options: {
          responsive: true, maintainAspectRatio: false,
          plugins: { legend: { labels: { color: '#71717a', font: { size: 10 } } } },
          scales: {
            x: { stacked: true, ticks: { color: '#52525b', font: { size: 9 }, maxRotation: 45 }, grid: { color: '#1f2028' } },
            y: { stacked: true, ticks: { color: '#52525b', font: { size: 9 }, callback: v => fmtNum(v) }, grid: { color: '#1f2028' } }
          }
        }
      });
    }
    updateDailyChart();

    // Model chart
    const models = Object.keys(d.by_model);
    const modelTotals = models.map(k => d.by_model[k].total);
    const shortModels = models.map(m => m.length > 22 ? m.slice(0, 22) + '…' : m);
    const colors = ['#667eea','#22c55e','#f5576c','#facc15','#a78bfa','#06b6d4','#f97316','#ec4899','#84cc16','#14b8a6'];

    if (!chartModels) {
      const ctx2 = document.getElementById('chart-models').getContext('2d');
      chartModels = new Chart(ctx2, {
        type: 'doughnut',
        data: {
          labels: shortModels,
          datasets: [{ data: modelTotals, backgroundColor: colors, borderWidth: 0 }]
        },
        options: {
          responsive: true, maintainAspectRatio: false,
          plugins: {
            legend: { position: 'right', labels: { color: '#71717a', font: { size: 9 }, boxWidth: 10 } },
            tooltip: { callbacks: { label: ctx => ctx.label + ': ' + fmtNum(ctx.raw) } }
          }
        }
      });
    } else {
      chartModels.data.labels = shortModels;
      chartModels.data.datasets[0].data = modelTotals;
      chartModels.update();
    }
  } catch(e) {}
}

// Hook into GPU refresh to update cost
const _origRefreshGpu = refreshGpu;
async function refreshGpu() {
  try {
    const r = await fetch('/api/gpu');
    const g = await r.json();
    if (g.error) return;
    currentPowerW = g.power;
    const vramPct = (g.vram_used / g.vram_total * 100);
    const vramGB = (g.vram_used / 1024).toFixed(1);
    const vramTotalGB = (g.vram_total / 1024).toFixed(0);
    document.getElementById('gpu-name').textContent = g.name;
    document.getElementById('gpu-vram').textContent = vramGB + ' GB';
    document.getElementById('gpu-vram-sub').textContent = 'of ' + vramTotalGB + ' GB (' + vramPct.toFixed(0) + '%)';
    document.getElementById('gpu-vram-bar').style.width = vramPct + '%';
    document.getElementById('gpu-util').textContent = g.gpu_util + '%';
    document.getElementById('gpu-util-bar').style.width = g.gpu_util + '%';
    const tempColor = g.temp > 80 ? '#ef4444' : g.temp > 65 ? '#facc15' : '#22c55e';
    document.getElementById('gpu-temp').textContent = g.temp + '\u00b0C';
    document.getElementById('gpu-temp-bar').style.width = (g.temp / 100 * 100) + '%';
    document.getElementById('gpu-temp-bar').style.background = tempColor;
    document.getElementById('gpu-power').textContent = g.power.toFixed(0) + ' W';
    document.getElementById('gpu-power-sub').textContent = 'of ' + g.power_limit.toFixed(0) + ' W';
    document.getElementById('gpu-power-bar').style.width = (g.power / g.power_limit * 100) + '%';
    updateCost();
  } catch(e) {}
}

refresh();
refreshGpu();
refreshTokens();
refreshPower();
setInterval(refresh, 5000);
setInterval(refreshGpu, 3000);
setInterval(refreshTokens, 60000);
setInterval(refreshPower, 60000);
setInterval(() => { if (allLogsOpen) updateAllLogs(); }, 3000);
</script>
</body>
</html>
"""

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=80)
