# tests/test_models_local.py
import json
import dashboard as d


def test_models_local_missing_noop(tmp_path, scratch):
    before = dict(d.MODELS)
    d._apply_models_local(str(tmp_path / "does_not_exist.json"))
    assert d.MODELS == before


def test_models_local_remove_and_add(tmp_path, scratch):
    p = tmp_path / "models_local.json"
    p.write_text(json.dumps({
        "remove": ["janus"],
        "add": {
            "probe_local": {"name": "Probe", "port": 8199},
            "bad_entry": {"no": "port"},
        },
    }))
    assert "janus" in d.MODELS
    d._apply_models_local(str(p))
    assert "janus" not in d.MODELS
    assert d.MODELS["probe_local"] == {"name": "Probe", "port": 8199}
    assert "bad_entry" not in d.MODELS


def test_models_local_corrupt_keeps_builtins(tmp_path, scratch):
    p = tmp_path / "models_local.json"
    p.write_text("{not json")
    before = dict(d.MODELS)
    d._apply_models_local(str(p))
    assert d.MODELS == before
