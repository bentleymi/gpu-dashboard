# tests/test_builder_scan.py
import dashboard as d


def test_families_cover_seeded():
    for fid in ["qwen38-27b", "qwen3-coder-next", "qwen36-35b-uncensored",
                "davidau-40b", "qwen35-122b", "minimax-m25",
                "qwen36-27b-fable-fusion", "qwen36-27b-fable-amd",
                "ornith-35b", "laguna-s21", "qwen36-27b", "qwen36-35b",
                "qwen25-72b", "qwen25-coder-32b", "llama33-70b",
                "deepseek-r1-32b", "deepseek-r1-70b", "qwen35-27b-opus-reasoning",
                "qwen38-flash-next"]:
        assert fid in d.MODEL_FAMILIES, fid


def test_qwen38_gguf_scan_real():
    variants = d.scan_gguf_variants("qwen38-27b")
    quants = sorted(v["quant"] for v in variants)
    assert "Q8_K_XL" in quants and "Q6_K_XL" in quants
    assert not any("mtp" in v["path"].lower() for v in variants)
    assert all(v["engine"] == "llama.cpp" for v in variants)


def test_shards_grouped_one_variant(tmp_path):
    vdir = tmp_path / "Q5_K_M"
    vdir.mkdir()
    for i in (1, 2, 3):
        (vdir / f"Qwen_Qwen3.5-122B-A10B-Q5_K_M-0000{i}-of-00003.gguf").write_bytes(b"x" * 10)
    vs = d._scan_gguf_variants_in(str(tmp_path), "llama.cpp", [])
    assert len(vs) == 1
    assert vs[0]["quant"] == "Q5_K_M"
    assert vs[0]["path"].endswith("00001-of-00003.gguf")
    assert vs[0]["label"].endswith("shards")


def test_two_quants_one_dir(tmp_path):
    for n in ("Model-Q6_K.gguf", "Model-HIGH-Q8_0.gguf"):
        (tmp_path / n).write_bytes(b"x" * 5)
    vs = d._scan_gguf_variants_in(str(tmp_path), "llama.cpp", [])
    assert sorted(v["quant"] for v in vs) == ["Q6_K", "Q8_0"]


def test_exclude_dir_skipped(tmp_path):
    (tmp_path / "main").mkdir()
    (tmp_path / "main" / "A-Q8_K_XL.gguf").write_bytes(b"x" * 5)
    (tmp_path / "mtp").mkdir()
    (tmp_path / "mtp" / "A-Q4_0.gguf").write_bytes(b"x" * 5)
    vs = d._scan_gguf_variants_in(str(tmp_path), "llama.cpp", ["mtp"])
    assert [v["quant"] for v in vs] == ["Q8_K_XL"]


def test_fable_amd_exact_real():
    vs = d.scan_gguf_variants("qwen36-27b-fable-amd")
    assert [v["quant"] for v in vs] == ["IQ4_XS"]


def test_flash_next_scan_and_engine_real():
    vs = d.scan_gguf_variants("qwen38-flash-next")
    assert sorted(v["quant"] for v in vs) == ["IQ4_XS", "Q4_K_XL"]
    assert all(v["engine"] == "llama.cpp" for v in vs)
    fam = d.MODEL_FAMILIES["qwen38-flash-next"]
    assert fam["engines"]["llama.cpp"]["bin"] == d.PR27742_LLAMA


def test_templates_real_skip_broken():
    names = d.scan_templates("qwen38-27b")
    assert "sharp-chat-template-v22.1.1.jinja" in names
    assert "qwen38-safe-v2.jinja" in names
    assert not any("broken" in n for n in names)


def test_sources_availability_real():
    by_id = {s["id"]: s for s in d.scan_sources("qwen36-27b")}
    assert by_id["bf16"]["available"] is True
    assert by_id["nvfp4"]["available"] is True
    assert by_id["nvfp4"]["engine"] == "vllm-nvfp4"
    assert by_id["bf16"]["ctx_options"][0]["value"] == 131072


def test_env_paths_point_at_scratch():
    assert d.CUSTOM_MODELS_FILE == "/tmp/builder-tests/custom_models.json"
    assert d.OPENCODE_CONFIG == "/tmp/builder-tests/opencode_config.json"


def test_no_family_references_llama_cpp_server():
    for fid, fam in d.MODEL_FAMILIES.items():
        assert "llama_cpp.server" not in fam.get("engines", {}), fid
