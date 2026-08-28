# tests/test_builder_opencode.py
import glob
import json
import os
import pytest
import dashboard as d


def cfg():
    with open(os.environ["OPENCODE_CONFIG_PATH"]) as f:
        return json.load(f)


def test_patch_creates_provider_and_model(scratch, real_opencode_copy):
    before = set(cfg()["provider"])
    warn = d.opencode_patch("myalias", 8100, "My Model", 262144, {"temperature": 0.6})
    assert warn is None
    c = cfg()
    p = c["provider"]["myalias-8100"]
    assert p["npm"] == "@ai-sdk/openai-compatible"
    assert p["options"] == {"baseURL": "http://localhost:8100/v1", "apiKey": "local",
                            "setCacheKey": False, "timeout": False}
    assert p["models"]["myalias"] == {
        "name": "My Model",
        "limit": {"context": 262144, "output": 32768},
        "options": {"temperature": 0.6}}
    assert before <= set(c["provider"])             # nothing removed
    assert glob.glob(os.environ["OPENCODE_CONFIG_PATH"] + ".bak-*")


def test_patch_reuses_provider_with_same_port(scratch, real_opencode_copy):
    c = cfg()
    cand = [pid for pid, p in c["provider"].items()
            if str((p.get("options") or {}).get("baseURL", "")).startswith("http")]
    if not cand:
        pytest.skip("real config has no local provider to reuse")
    pid = cand[0]
    port = int(c["provider"][pid]["options"]["baseURL"].rsplit(":", 1)[1].split("/")[0])
    assert d.opencode_patch("aliasB", port, "B", 131072, {}) is None
    c2 = cfg()
    assert f"aliasB-{port}" not in c2["provider"]
    assert "aliasB" in c2["provider"][pid]["models"]
    assert "options" not in c2["provider"][pid]["models"]["aliasB"]


def test_missing_file_returns_warning_and_does_not_create(scratch):
    warn = d.opencode_patch("a", 8100, "A", 4096, {})
    assert warn and "not found" in warn
    assert not os.path.exists(os.environ["OPENCODE_CONFIG_PATH"])


def test_unpatch_removes_model_then_empty_provider(scratch, real_opencode_copy):
    before = cfg()
    d.opencode_patch("myalias", 8100, "My Model", 262144, {})
    d.opencode_unpatch("myalias-8100", "myalias")
    after = cfg()
    assert "myalias-8100" not in after["provider"]
    for pid in before["provider"]:
        assert after["provider"].get(pid) == before["provider"][pid], pid
    # top-level keys preserved
    for k in before:
        if k != "provider":
            assert after.get(k) == before[k], k


def test_backup_rotation_keeps_10(scratch, real_opencode_copy):
    base = os.environ["OPENCODE_CONFIG_PATH"]
    for i in range(1, 13):
        with open(f"{base}.bak-2026{i:02d}01-000000", "w") as f:
            json.dump({}, f)
    d.opencode_patch("rot", 8100, "R", 4096, {})
    assert len(glob.glob(base + ".bak-*")) == 10
