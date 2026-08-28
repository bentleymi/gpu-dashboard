# tests/test_builder_api.py
import json
import os
import dashboard as d

B = {"family": "qwen38-27b", "variant": "gguf_q8_k_xl", "ctx": 262144}


def test_families_shape(client, scratch):
    r = client.get("/api/families")
    assert r.status_code == 200
    fams = {f["id"]: f for f in r.json()}
    assert "qwen38-27b" in fams and "qwen38-flash-next" in fams
    f38 = fams["qwen38-27b"]
    v8 = [v for v in f38["variants"] if v["id"] == "gguf_q8_k_xl"][0]
    assert v8["available"] is True and v8["quant"] == "Q8_K_XL"
    assert v8["ctx_options"][0]["value"] == 262144
    assert "sharp-chat-template-v22.1.1.jinja" in f38["templates"]
    assert f38["kv_cache"] == ["f16", "q8_0", "q4_0"]
    assert f38["sampling_defaults"]["llama.cpp"] == \
        {"temp": 0.8, "top_p": 0.95, "repeat_penalty": 1.1}
    assert fams["qwen36-27b"]["kv_cache"] == []      # vLLM-only family
    fn = fams["qwen38-flash-next"]
    assert sorted(v["quant"] for v in fn["variants"]) == ["IQ4_XS", "Q4_K_XL"]


def test_post_validation(client, scratch):
    r = client.post("/api/custom-model", json={"family": "nope"})
    assert r.status_code == 400 and r.json()["field"] == "family"
    r = client.post("/api/custom-model", json=dict(B, variant="nope"))
    assert r.status_code == 400 and r.json()["field"] == "variant"
    r = client.post("/api/custom-model", json=dict(B, ctx=123))
    assert r.status_code == 400 and r.json()["field"] == "ctx"


def test_post_save_and_delete_cycle(client, scratch, monkeypatch):
    monkeypatch.setattr(d, "is_port_open", lambda p: False)
    body = dict(B, advanced={"temp": 0.6, "enable_thinking": False})
    r = client.post("/api/custom-model", json=body)
    assert r.status_code == 200, r.text
    e = r.json()["entry"]
    assert e["id"] == "cust_qwen38_27b_gguf_q8_k_xl_256k"
    assert e["port"] == 8100
    assert e["custom"] is True
    assert e["id"] in d.MODELS
    # second (different variant) → 8101; duplicate → 400
    r2 = client.post("/api/custom-model",
                     json={"family": "qwen38-27b", "variant": "gguf_q6_k_xl", "ctx": 262144})
    assert r2.json()["entry"]["port"] == 8101
    r3 = client.post("/api/custom-model", json=body)
    assert r3.status_code == 400
    assert "already exists as cust_qwen38_27b_gguf_q8_k_xl_256k" in r3.json()["message"]
    # delete → gone from MODELS + JSON; next save reuses 8100
    r4 = client.delete(f"/api/custom-model/{e['id']}")
    assert r4.status_code == 200
    assert e["id"] not in d.MODELS
    with open(os.environ["DASHBOARD_CUSTOM_MODELS"]) as f:
        assert [x["id"] for x in json.load(f)] == ["cust_qwen38_27b_gguf_q6_k_xl_256k"]
    r5 = client.post("/api/custom-model", json=body)
    assert r5.json()["entry"]["port"] == 8100


def test_delete_builtin_404(client, scratch):
    r = client.delete("/api/custom-model/vllm_qwen38_27b")
    assert r.status_code == 404
