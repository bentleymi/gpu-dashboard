# tests/test_builder_resolve.py
import dashboard as d

V8 = {v["id"]: v for v in d.family_variants("qwen38-27b")}["gguf_q8_k_xl"]


def adv(**kw):
    base = {"ctx": 262144}
    base.update(kw)
    return base


def test_known_family_defaults_emitted():
    res, errs, warns = d.resolve_advanced("qwen38-27b", V8, adv())
    assert errs == []
    assert res["reasoning"] == "medium"
    assert res["thinking"] is True
    assert res["kv"] is None                 # f16 default
    assert res["yarn_orig"] is None
    assert res["temp"] is None               # no sampling flag unless user set one
    assert res["vram"] == 46                 # per_variant[Q8_K_XL]
    assert res["template_path"].endswith("sharp-chat-template-v22.1.1.jinja")


def test_user_values_override():
    res, errs, _ = d.resolve_advanced("qwen38-27b", V8,
                                      adv(reasoning_effort="high",
                                          enable_thinking=False, temp=0.6))
    assert errs == []
    assert res["reasoning"] == "high"
    assert res["thinking"] is False
    assert res["temp"] == 0.6


def test_off_means_no_flag():
    res, errs, _ = d.resolve_advanced("qwen38-27b", V8, adv(reasoning_effort="off"))
    assert errs == [] and res["reasoning"] is None


def test_unknown_family_emits_only_explicit():
    vid = d.family_variants("qwen36-27b-fable-fusion")[0]["id"]
    v = {x["id"]: x for x in d.family_variants("qwen36-27b-fable-fusion")}[vid]
    res, errs, warns = d.resolve_advanced("qwen36-27b-fable-fusion", v, adv())
    assert errs == [] and res["reasoning"] is None and res["thinking"] is None
    res, errs, warns = d.resolve_advanced("qwen36-27b-fable-fusion", v,
                                          adv(reasoning_effort="high", enable_thinking=True))
    assert errs == [] and res["reasoning"] == "high" and res["thinking"] is True
    assert any("not verified" in w for w in warns)


def test_hidden_fields_reject_values():
    v = d.family_variants("qwen36-27b-fable-amd")[0]          # thinking hidden
    res, errs, _ = d.resolve_advanced("qwen36-27b-fable-amd", v, adv(enable_thinking=True))
    assert any(e["field"] == "enable_thinking" for e in errs)
    v2 = d.family_variants("qwen3-coder-next")[0]             # reasoning unsupported
    res, errs, _ = d.resolve_advanced("qwen3-coder-next", v2, adv(reasoning_effort="high"))
    assert any(e["field"] == "reasoning_effort" for e in errs)


def test_kv_rules():
    v = d.family_variants("qwen36-35b-uncensored")[0]
    res, errs, _ = d.resolve_advanced("qwen36-35b-uncensored", v, adv(ctx=1048576))
    assert errs == []
    assert res["kv"] == "q8_0"            # 1M option's kv_default
    assert res["yarn_orig"] == 262144
    v36 = [x for x in d.family_variants("qwen36-27b") if x["engine"] == "vllm"][0]
    res, errs, _ = d.resolve_advanced("qwen36-27b", v36, adv(ctx=131072, kv_cache="q8_0"))
    assert any(e["field"] == "kv_cache" for e in errs)


def test_ctx_custom_and_invalid():
    res, errs, warns = d.resolve_advanced("qwen38-27b", V8, adv(ctx=999999))
    assert errs == [] and res["custom_ctx"] is True
    assert any("not a verified length" in w for w in warns)
    res, errs, _ = d.resolve_advanced("qwen38-27b", V8, {})
    assert any(e["field"] == "ctx" for e in errs)
    assert res == {}


def test_sampling_ranges():
    res, errs, _ = d.resolve_advanced("qwen38-27b", V8, adv(temp=0, top_p=1.0, repeat_penalty=0.1))
    assert errs == [] and res["top_p"] == 1.0 and res["repeat_penalty"] == 0.1
    res, errs, _ = d.resolve_advanced("qwen38-27b", V8, adv(top_p=0))
    assert any(e["field"] == "top_p" for e in errs)
    res, errs, _ = d.resolve_advanced("qwen38-27b", V8, adv(temp="x"))
    assert any(e["field"] == "temp" for e in errs)
