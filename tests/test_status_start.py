# tests/test_status_start.py
import dashboard as d


def test_is_vllm_cmd_forms():
    assert d.is_vllm_cmd(["/x/venv/bin/vllm", "serve", "m"]) is True
    assert d.is_vllm_cmd(["/x/venv/bin/python", "-m", "vllm.entrypoints.openai.api_server"]) is True
    assert d.is_vllm_cmd(["/x/venv/bin/python", "-m", "vllm.serve"]) is True
    assert d.is_vllm_cmd(["/llama-server", "--model", "m"]) is False
    assert d.is_vllm_cmd(["/x/venv/bin/python", "app.py"]) is False
    assert d.is_vllm_cmd([]) is False
    assert d.is_vllm_cmd(["/x/venv/bin/python", "-m"]) is False


def test_status_custom_flag(client, scratch):
    d.MODELS["cust_probe"] = {
        "name": "Probe", "description": "", "port": 8150, "cmd": ["x"], "cwd": "/",
        "env": {}, "protocol": "http", "category": "LLM", "icon": "i", "color": "#fff",
        "tags": [], "supports_offload": False, "custom": True}
    j = client.get("/api/status").json()
    assert j["cust_probe"]["custom"] is True
    assert j["vllm_qwen38_27b"]["custom"] is False
