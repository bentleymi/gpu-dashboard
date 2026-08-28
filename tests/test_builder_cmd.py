# tests/test_builder_cmd.py
import json
import dashboard as d

V8 = {v["id"]: v for v in d.family_variants("qwen38-27b")}["gguf_q8_k_xl"]


def res_for(fid, v, ctx=262144, **adv):
    a = {"ctx": ctx}
    a.update(adv)
    res, errs, _ = d.resolve_advanced(fid, v, a)
    assert not errs, errs
    return res


def test_make_alias():
    assert d.make_alias("qwen38-27b", V8, 262144) == "qwen38_27b_gguf_q8_k_xl_256k"


def test_llama_golden_cmd_defaults():
    res = res_for("qwen38-27b", V8)
    cmd, env = d.build_launch_cmd("qwen38-27b", V8, res, 8100, "qwen38_27b_gguf_q8_k_xl_256k")
    assert env == {}
    assert cmd == [d.QWEN38_LLAMA,
                   "--model", V8["path"],
                   "--alias", "qwen38_27b_gguf_q8_k_xl_256k",
                   "--ctx-size", "262144",
                   "-ngl", "99", "-b", "2048", "-ub", "2048",
                   "--host", "0.0.0.0", "--port", "8100", "--threads", "16",
                   "--chat-template-file",
                   "/mnt/raid1_nvme/Qwen3_8-27B/templates/sharp-chat-template-v22.1.1.jinja",
                   "--spec-draft-model",
                   "/mnt/raid1_nvme/Qwen3_8-27B/gguf-q8/mtp/MTP/mtp-Qwen3.8-27B-Q4_0.gguf",
                   "--spec-type", "draft-mtp", "--spec-draft-ngl", "99",
                   "--chat-template-kwargs", '{"enable_thinking": true}',
                   "--reasoning-effort", "medium"]


def test_llama_yarn_kv_jinja_sampling():
    v = d.family_variants("qwen36-35b-uncensored")[0]
    res = res_for("qwen36-35b-uncensored", v, ctx=1048576, temp=0.7, top_p=0.9)
    cmd, _ = d.build_launch_cmd("qwen36-35b-uncensored", v, res, 8101, "x")
    assert cmd[:2] == [d.IK_LLAMA, "--model"]
    i = cmd.index("--rope-scaling")
    assert cmd[i:i + 4] == ["--rope-scaling", "yarn", "--yarn-orig-ctx", "262144"]
    assert cmd[i + 4:i + 8] == ["-ctk", "q8_0", "-ctv", "q8_0"]
    assert "--jinja" in cmd                        # no template dir, engine jinja=True
    assert "--chat-template-file" not in cmd
    assert cmd[-4:] == ["--temp", "0.7", "--top-p", "0.9"]
    assert "--repeat-penalty" not in cmd
    assert "--reasoning-effort" not in cmd         # unknown + unset
    # thinking is KNOWN (default false) for this family -> always emitted
    kw = cmd[cmd.index("--chat-template-kwargs") + 1]
    assert json.loads(kw) == {"enable_thinking": False}
    assert "--spec-draft-model" not in cmd         # family has none


def test_flash_next_uses_pr27742_binary():
    v = [x for x in d.family_variants("qwen38-flash-next") if x["quant"] == "IQ4_XS"][0]
    res = res_for("qwen38-flash-next", v)
    cmd, _ = d.build_launch_cmd("qwen38-flash-next", v, res, 8102, "fn")
    assert cmd[0] == d.PR27742_LLAMA
    assert res["vram"] == 94


def test_vllm_golden_cmd_and_env():
    v = [x for x in d.family_variants("qwen36-27b") if x["engine"] == "vllm"][0]
    res = res_for("qwen36-27b", v, ctx=131072, enable_thinking=True)
    cmd, env = d.build_launch_cmd("qwen36-27b", v, res, 8103, "x")
    assert cmd == [d.VLLM_LEGACY, "serve", "/mnt/raid1_nvme/models/qwen36-27b",
                   "--served-model-name", "x",
                   "--host", "0.0.0.0", "--port", "8103",
                   "--max-model-len", "131072",
                   "--gpu-memory-utilization", "0.90",
                   "--trust-remote-code",
                   "--default-chat-template-kwargs", '{"enable_thinking": true}']
    assert env == {"PYTORCH_ALLOC_CONF": "expandable_segments:True"}


def test_nvfp4_env_and_template():
    v = [x for x in d.family_variants("qwen38-27b") if x["engine"] == "vllm-nvfp4"][0]
    res = res_for("qwen38-27b", v)
    cmd, env = d.build_launch_cmd("qwen38-27b", v, res, 8104, "nv")
    assert cmd[:3] == [d.NVFP4_38, "serve", "unsloth/Qwen3.8-27B-NVFP4"]
    assert env == {"PYTORCH_ALLOC_CONF": "expandable_segments:True",
                   "FLASHINFER_DISABLE_VERSION_CHECK": "1"}
    assert "--chat-template" in cmd                # family default template is a file
    kw = cmd[cmd.index("--default-chat-template-kwargs") + 1]
    assert json.loads(kw) == {"enable_thinking": True, "reasoning_effort": "medium"}


def test_build_custom_entry_shape():
    res = res_for("qwen38-27b", V8, enable_thinking=False, temp=0.6)
    eid, e = d.build_custom_entry("qwen38-27b", V8, res, {"description": "", "tags": []}, 8100)
    assert eid == "cust_qwen38_27b_gguf_q8_k_xl_256k"
    assert e["id"] == eid
    assert e["name"] == "Qwen3.8 27B (Q8_K_XL, 256K ctx)"
    assert e["description"].startswith("Q8_K_XL")     # auto description from variant label
    assert e["port"] == 8100
    assert e["supports_offload"] is False             # llama.cpp
    assert e["vram_gb"] == 46
    assert e["quant"] == "Q8_K_XL"
    assert e["custom"] is True
    assert e["custom_ref"] == {"family": "qwen38-27b", "variant": "gguf_q8_k_xl", "ctx": 262144}
    assert e["opencode"] == {"provider": "qwen38_27b_gguf_q8_k_xl_256k-8100",
                             "model_id": "qwen38_27b_gguf_q8_k_xl_256k"}
    assert e["tags"] == ["text-to-text", "text-to-code"]   # family tags pre-filled
