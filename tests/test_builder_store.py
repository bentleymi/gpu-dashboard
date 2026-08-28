# tests/test_builder_store.py
import glob
import json
import os
import dashboard as d


def test_roundtrip(scratch):
    path = scratch[0]
    assert d.load_custom_entries(path) == []
    d.save_custom_entries(path, [{"id": "cust_a", "name": "A"}])
    assert d.load_custom_entries(path) == [{"id": "cust_a", "name": "A"}]


def test_corrupt_file_quarantined(scratch):
    path = scratch[0]
    with open(path, "w") as f:
        f.write("{not json")
    assert d.load_custom_entries(path) == []
    assert glob.glob(path + ".corrupt-*")
    assert not os.path.exists(path)


def test_merge_skips_builtin_collision(scratch):
    path = scratch[0]
    d.save_custom_entries(path,
                          [{"id": "vllm_qwen38_27b", "name": "steal", "description": "x",
                            "port": 1, "cmd": [], "cwd": "/", "env": {}, "protocol": "http",
                            "category": "LLM", "icon": "i", "color": "#fff", "tags": [],
                            "supports_offload": False},
                           {"id": "cust_ok", "name": "ok", "description": "x", "port": 2,
                            "cmd": [], "cwd": "/", "env": {}, "protocol": "http",
                            "category": "LLM", "icon": "i", "color": "#fff", "tags": [],
                            "supports_offload": False}])
    d.merge_custom_at_startup()
    assert d.MODELS["vllm_qwen38_27b"]["name"] != "steal"   # built-in preserved
    assert "cust_ok" in d.MODELS and "cust_ok" in d.CUSTOM_IDS


def test_alloc_port_first_free(scratch, monkeypatch):
    monkeypatch.setattr(d, "is_port_open", lambda p: p == 8100)
    assert d.alloc_port() == 8101
    monkeypatch.setattr(d, "is_port_open", lambda p: False)
    assert d.alloc_port() == 8100
    for p in range(8100, 8200):
        d.MODELS[f"blocker{p}"] = {"name": "b", "description": "", "port": p, "cmd": [],
                                   "cwd": "/", "env": {}, "protocol": "http", "category": "x",
                                   "icon": "x", "color": "#fff", "tags": [],
                                   "supports_offload": False}
    try:
        d.alloc_port()
        assert False, "expected RuntimeError"
    except RuntimeError as e:
        assert "8100-8199" in str(e)
