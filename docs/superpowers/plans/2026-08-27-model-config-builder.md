# Model Config Builder Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add a builder to the GPU dashboard that composes launch configs from model family + on-disk quant + known-safe context + advanced options, persists them as custom models, and mirrors them into `~/.config/opencode/config.json`.

**Architecture:** The single-file FastAPI app (`dashboard.py`) gains a `MODEL_FAMILIES` metadata table (curated capabilities) intersected with a live disk scan (available quants/weights), three new endpoints (`GET /api/families`, `POST/DELETE /api/custom-model`), a `custom_models.json` store merged into `MODELS` (startup + runtime), pure command-builder functions per engine, and a collapsible builder panel in the existing `HTML_PAGE`. Custom rows render through the unchanged `/api/status` pipeline plus one `custom` flag.

**Tech Stack:** Python 3.12 / FastAPI (already in `dashboard-venv`), stdlib-only runtime code; `pytest` + `httpx` (dev-only in the venv); vanilla JS inside `HTML_PAGE`; `node --check` for JS syntax.

**Spec:** `/mnt/raid1_nvme/JanusPro7b/docs/superpowers/specs/2026-08-26-model-config-builder-design.md` — read it alongside this plan; the spec owns the *what*, this plan owns the *how*.

## Verified facts (measured on this machine during planning)

- `dashboard-venv` has fastapi 0.135.0 but **no pytest/httpx** — Task 1 installs them (dev-only).
- `node v20.20.2` available. All engine binaries exist: ik `llama-server`, Qwen3.8 `llama-server`,
  the llama.cpp PR 27742 build `/mnt/raid1_nvme/models/llama.cpp-pr27742/build/bin/llama-server`
  (family `qwen38-flash-next`; its GGUF arch is only supported by that build),
  `/mnt/raid1_nvme/Qwen3_8-27B/venv/bin/vllm`, `/mnt/raid1_sata/vllm-env/bin/vllm`,
  both NVFP4 venvs.
- Existing builtins: every `llama-server` entry has `supports_offload: False`; vLLM entries
  `supports_offload: True`. The qwen38 vLLM entries (8010/8017) launch
  `.../venv/bin/python -m vllm.entrypoints.openai.api_server` (module form — why the
  `is_vllm_cmd` fix exists) and carry `--chat-template <sharp jinja>` +
  `--default-chat-template-kwargs '{"enable_thinking": true, "reasoning_effort": "medium"}'`.
- opencode config shape (top-level keys `$schema, model, plugin, provider`):
  `provider.<pid> = {npm: "@ai-sdk/openai-compatible", name, options: {baseURL, apiKey, setCacheKey: false, timeout: false}, models: {<model_id>: {name, limit: {context, output}}}}`
  — `setCacheKey`/`timeout` live **inside `options`**. Model-level `options: {}` blocks are
  currently unused in the real config.
- opencode binary (`/root/.opencode/bin/opencode`) contains the strings `repetitionPenalty` and
  `repetition_penalty` — we write `repetitionPenalty` (Task 10 documents a manual runtime check).
- `/models` is a symlink (older dashboard entries use `/models/...` paths; new code always uses
  the real `/mnt/raid1_nvme/models/...` paths).

## Spec erratum (verified on disk, approved direction)

Spec §3 listed a `llama_cpp.server` engine for `ornith-35b`/`laguna-s21`. Those venvs contain **no
`llama_cpp` module** (`import llama_cpp` → ModuleNotFoundError), so the engine can't work. Both
GGUF arches (`qwen35moe` — read from the Ornith GGUF header; `laguna` — in
`ik_llama.cpp/src/llama-arch.cpp`) are supported by the **ik llama.cpp binary**, so **all GGUF
families use the `llama.cpp` engine** (ik build; qwen3.8 build for `qwen38-27b`).
`llama_cpp.server` is dropped from v1. The pre-existing (broken) ornith/laguna dashboard entries are
out of scope — do not touch them.

## Global Constraints

- NEVER start/stop/restart anything that consumes VRAM: no `systemctl restart gpu-dashboard.service`,
  no model launches in tests. Test instances bind **127.0.0.1 only**, ports 8098/8099.
- Tests never touch real config files: paths come from env overrides `DASHBOARD_CUSTOM_MODELS`
  (default `/mnt/raid1_nvme/JanusPro7b/custom_models.json`) and `OPENCODE_CONFIG_PATH`
  (default `/root/.config/opencode/config.json`).
- Custom ports allocated only from **8100–8199**, first free (not in `MODELS`, not listening).
- Advanced field blank ⇒ engine/family default. For *unknown* capability families, no flag is ever
  emitted unless the user explicitly sets the field (spec §3 emit rules).
- opencode config: back up to `config.json.bak-<YYYYmmdd-HHMMSS>` (keep last 10) before every
  write; atomic writes (`tmp` + `os.replace`); **never create the file if it doesn't exist**.
- No new runtime dependencies for the app (stdlib + FastAPI). `pytest`/`httpx` are dev-only.
- `.gitignore` is `*` + `!.gitignore`: every new tracked file needs `git add -f` (repo convention).
- Commit after every task and push. Short imperative subject + bullet body (matches repo history).

## File Structure

- **Modify** `/mnt/raid1_nvme/JanusPro7b/dashboard.py`:
  - New `# ── Model Config Builder ──` section inserted immediately after the `op_lock = threading.Lock()`
    line (1265), before `app = FastAPI()`: env-path constants, `MODEL_FAMILIES`, scan helpers,
    `resolve_advanced`, `build_launch_cmd`, `build_custom_entry`, store + `alloc_port`, opencode
    patcher, and the `merge_custom_at_startup()` call.
   - Three endpoints inserted after `api_logs` (before `@app.get("/")`, currently ~line 1690);
     `is_vllm_cmd` helper added before `api_start` + offload-detection fix inside it
     (currently ~line 1575); `"custom"` key in `api_status` (currently ~line 1546).
     NOTE: everything below the builder-section insertion shifts by the section's size — the
     code anchors (function names, the `@app.get("/")` line) are authoritative, not the numbers.
   - Frontend: CSS appended before `</style>`; builder panel HTML before `<div class="table-bar">`
     (currently ~line 2152); JS functions appended before the `refresh();` bootstrap
     (currently ~line 2625); `buildRow` name cell extended with custom badge + trash button.
- **Create** `tests/conftest.py`, `tests/test_smoke.py`, `tests/test_builder_scan.py`,
  `tests/test_builder_resolve.py`, `tests/test_builder_cmd.py`, `tests/test_builder_store.py`,
  `tests/test_builder_opencode.py`, `tests/test_builder_api.py`, `tests/test_status_start.py`,
  `tests/test_page.py`.
- `custom_models.json` is created at runtime on first real save (never committed).

---

### Task 1: Test infrastructure

**Files:**
- Create: `tests/conftest.py`, `tests/test_smoke.py`
- Modify: venv (dev deps only)

**Interfaces:**
- Produces: pytest fixtures `client` (`TestClient` on `dashboard.app`) and `scratch` (isolated
  config files + pristine `MODELS`/`CUSTOM_IDS` restore); env isolation for
  `DASHBOARD_CUSTOM_MODELS` / `OPENCODE_CONFIG_PATH` (set to `/tmp/builder-tests/...`).
- Consumes: nothing.

- [ ] **Step 1: Install dev deps**

```bash
# NOTE: dashboard-venv/bin/pip has a stale shebang (/mnt/raid1_sata/...) — use -m pip
/mnt/raid1_nvme/JanusPro7b/dashboard-venv/bin/python -m pip install -q pytest httpx
/mnt/raid1_nvme/JanusPro7b/dashboard-venv/bin/python -c "import pytest, httpx; print('dev deps ok')"
```

- [ ] **Step 2: Write `tests/conftest.py`**

```python
import os
import shutil
import sys

SCRATCH = "/tmp/builder-tests"
os.makedirs(SCRATCH, exist_ok=True)
os.environ["DASHBOARD_CUSTOM_MODELS"] = os.path.join(SCRATCH, "custom_models.json")
os.environ["OPENCODE_CONFIG_PATH"] = os.path.join(SCRATCH, "opencode_config.json")

sys.path.insert(0, "/mnt/raid1_nvme/JanusPro7b")

import pytest
from fastapi.testclient import TestClient

import dashboard  # noqa: E402  (import AFTER env vars are set)

REAL_OPENCODE = "/root/.config/opencode/config.json"


@pytest.fixture
def client():
    return TestClient(dashboard.app)


@pytest.fixture
def scratch():
    """Fresh scratch files + pristine MODELS/custom registry for one test."""
    paths = (os.environ["DASHBOARD_CUSTOM_MODELS"], os.environ["OPENCODE_CONFIG_PATH"])
    for p in paths:
        if os.path.exists(p):
            os.remove(p)
    models_backup = dict(dashboard.MODELS)
    ids_backup = set(getattr(dashboard, "CUSTOM_IDS", set()))
    yield paths
    dashboard.MODELS.clear()
    dashboard.MODELS.update(models_backup)
    dashboard.CUSTOM_IDS = ids_backup
    for p in paths:
        if os.path.exists(p):
            os.remove(p)


@pytest.fixture
def real_opencode_copy():
    """Copy the real opencode config into the scratch path (shape-faithful)."""
    dst = os.environ["OPENCODE_CONFIG_PATH"]
    shutil.copy(REAL_OPENCODE, dst)
    return dst
```

- [ ] **Step 3: Write `tests/test_smoke.py`**

```python
def test_status_endpoint(client):
    r = client.get("/api/status")
    assert r.status_code == 200
    assert "vllm_qwen38_27b" in r.json()
```

- [ ] **Step 4: Run**

Run: `cd /mnt/raid1_nvme/JanusPro7b && dashboard-venv/bin/python -m pytest tests/test_smoke.py -v`
Expected: PASS (exercises only existing behavior; if the import or fixture breaks, fix conftest
before continuing — every later task builds on it).

- [ ] **Step 5: Commit**

```bash
git -C /mnt/raid1_nvme/JanusPro7b add -f tests/conftest.py tests/test_smoke.py
git -C /mnt/raid1_nvme/JanusPro7b commit -m "Add builder test infrastructure (pytest, env-isolated scratch configs)"
git -C /mnt/raid1_nvme/JanusPro7b push
```

---

### Task 2: Family metadata + disk scan

**Files:**
- Modify: `dashboard.py` — new section after `op_lock = threading.Lock()` (line 1265)
- Create: `tests/test_builder_scan.py`

**Interfaces:**
- Produces:
  - `CUSTOM_MODELS_FILE: str`, `OPENCODE_CONFIG: str` (env-overridable paths)
  - `MODEL_FAMILIES: dict[str, dict]` (seeded table below; keys per spec §3)
  - `parse_quant(filename: str) -> str`
  - `_scan_gguf_variants_in(root: str, engine_name: str, exclude: list) -> list[dict]`
  - `scan_gguf_variants(fid: str) -> list[dict]` — variant dicts
    `{id, quant, path, weights_gb, engine, kind: "gguf", available: True, label, ctx_options: None}`
  - `scan_sources(fid: str) -> list[dict]` — `kind: "dir"|"hf"`, `ctx_options` = source override
    or `None`, dir availability = dir contains `*.safetensors`
  - `family_variants(fid: str) -> list[dict]`
  - `scan_templates(fid: str) -> list[str]` (`*.jinja`, names containing `broken` excluded)
- Consumes: stdlib only.

- [ ] **Step 1: Write failing tests**

```python
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
```

- [ ] **Step 2: Run to verify they fail**

Run: `cd /mnt/raid1_nvme/JanusPro7b && dashboard-venv/bin/python -m pytest tests/test_builder_scan.py -v`
Expected: FAIL `AttributeError: module 'dashboard' has no attribute 'MODEL_FAMILIES'`.

- [ ] **Step 3: Implement the builder section header + metadata**

Insert after `op_lock = threading.Lock()`. NOTE: `dashboard.py` has no bare `import json` at
module level (only aliased inside functions) — the builder section starts with `import json`.

```python
# ── Model Config Builder ─────────────────────────────────────────
import json
import shutil
import sys
import datetime

CUSTOM_MODELS_FILE = os.environ.get(
    "DASHBOARD_CUSTOM_MODELS", "/mnt/raid1_nvme/JanusPro7b/custom_models.json")
OPENCODE_CONFIG = os.environ.get(
    "OPENCODE_CONFIG_PATH", "/root/.config/opencode/config.json")

IK_LLAMA = "/mnt/raid1_nvme/models/ik_llama.cpp/build/bin/llama-server"
QWEN38_LLAMA = "/mnt/raid1_nvme/Qwen3_8-27B/llama.cpp/build/bin/llama-server"
VLLM_MAIN = "/mnt/raid1_nvme/Qwen3_8-27B/venv/bin/vllm"
VLLM_LEGACY = "/mnt/raid1_sata/vllm-env/bin/vllm"
NVFP4_36 = "/mnt/raid1_nvme/models/Qwen3.6-27B-NVFP4/venv/bin/vllm"
NVFP4_38 = "/mnt/raid1_nvme/models/Qwen3.8-27B-NVFP4/venv/bin/vllm"
PR27742_LLAMA = "/mnt/raid1_nvme/models/llama.cpp-pr27742/build/bin/llama-server"
FLASH_ENV = {"FLASHINFER_DISABLE_VERSION_CHECK": "1"}
PYTORCH_ENV = {"PYTORCH_ALLOC_CONF": "expandable_segments:True"}
IK_CWD = "/mnt/raid1_nvme/models/ik_llama.cpp"


def _fam(name, icon, color, tags, **kw):
    e = {"name": name, "icon": icon, "color": color, "tags": tags,
         "sources": [], "ctx_options": [], "docs_url": None, "scan_exclude": [],
         "spec_draft": None, "templates": [], "default_template": "builtin",
         "kv_cache": ["f16", "q8_0", "q4_0"],
         "reasoning": {"supported": "unknown"}, "thinking": {"toggleable": "unknown"}}
    e.update(kw)
    return e


def _llama(bin_=IK_LLAMA, cwd=IK_CWD, jinja=True):
    return {"bin": bin_, "cwd": cwd, "jinja": jinja}


def _vllm(binp, cwd, **kw):
    return {"vllm": binp, "cwd": cwd, **kw}


MODEL_FAMILIES = {
    "qwen38-27b": _fam(
        "Qwen3.8 27B", "sparkles", "#f97316", ["text-to-text", "text-to-code"],
        base_dir="/mnt/raid1_nvme/Qwen3_8-27B", scan_engine="llama.cpp",
        scan_exclude=["gguf-q8/mtp"],
        spec_draft="gguf-q8/mtp/MTP/mtp-Qwen3.8-27B-Q4_0.gguf",
        engines={
            "llama.cpp": _llama(QWEN38_LLAMA, "/mnt/raid1_nvme/Qwen3_8-27B", jinja=False),
            "vllm": _vllm(VLLM_MAIN, "/mnt/raid1_nvme/Qwen3_8-27B", dtype="half"),
            "vllm-nvfp4": _vllm(NVFP4_38, "/mnt/raid1_nvme/models/Qwen3.8-27B-NVFP4",
                                env=dict(FLASH_ENV)),
        },
        sources=[
            {"id": "fp16", "kind": "dir", "path": "/mnt/raid1_nvme/Qwen3_8-27B/fp16",
             "quant": "FP16", "engine": "vllm"},
            {"id": "uncensored", "kind": "dir",
             "path": "/mnt/raid1_nvme/Qwen3_8-27B/uncensored",
             "quant": "FP16", "engine": "vllm",
             "description": "orcarouter abliterated finetune"},
            {"id": "nvfp4", "kind": "hf", "path": "unsloth/Qwen3.8-27B-NVFP4",
             "quant": "NVFP4", "weights_gb": 16, "engine": "vllm-nvfp4"},
        ],
        ctx_options=[{"value": 262144, "label": "256K (native)", "kv_default": "f16",
                      "per_variant": {"Q8_K_XL": {"vram": 46}, "Q6_K_XL": {"vram": 40}}}],
        reasoning={"supported": True, "levels": ["low", "medium", "high"],
                   "default": "medium"},
        thinking={"toggleable": True, "default": True},
        templates=["templates"],
        default_template="templates/sharp-chat-template-v22.1.1.jinja"),
    "qwen3-coder-next": _fam(
        "Qwen3-Coder-Next", "code-2", "#10b981",
        ["text-to-text", "text-to-code", "agents"],
        base_dir="/mnt/raid1_nvme/models/qwen3-coder-next", scan_engine="llama.cpp",
        engines={"llama.cpp": _llama()},
        ctx_options=[{"value": 131072, "label": "131K (native)", "kv_default": "f16"}],
        reasoning={"supported": False}, thinking={"toggleable": False}),
    "qwen36-35b-uncensored": _fam(
        "Qwen3.6 35B Uncensored", "zap", "#f97316",
        ["text-to-text", "text-to-code", "agents"],
        base_dir="/mnt/raid1_nvme/models/qwen36-35b-uncensored", scan_engine="llama.cpp",
        engines={"llama.cpp": _llama()},
        ctx_options=[
            {"value": 262144, "label": "256K (native)", "kv_default": "f16"},
            {"value": 524288, "label": "512K (YaRN)", "yarn_orig": 262144,
             "kv_default": "f16", "note": "YaRN-extended; quality may degrade past native"},
            {"value": 1048576, "label": "1M (YaRN, q8 KV)", "yarn_orig": 262144,
             "kv_default": "q8_0", "vram": 84,
             "note": "YaRN-extended; needs most of VRAM"}],
        thinking={"toggleable": True, "default": False}),
    "davidau-40b": _fam(
        "Qwen3.6-40B Deck Opus", "brain", "#818cf8",
        ["text-to-text", "text-to-code", "agents", "claude"],
        base_dir="/mnt/raid1_nvme/models/davidau-qwen3.6-40b", scan_engine="llama.cpp",
        engines={"llama.cpp": _llama()},
        ctx_options=[
            {"value": 262144, "label": "256K (native)", "kv_default": "f16"},
            {"value": 393216, "label": "384K (YaRN, q4 KV)", "yarn_orig": 262144,
             "kv_default": "q4_0", "per_variant": {"Q8_0": {"vram": 81}},
             "note": "YaRN-extended"},
            {"value": 524288, "label": "512K (YaRN)", "yarn_orig": 262144,
             "kv_default": "q8_0",
             "per_variant": {"Q6_K": {"kv": "q4_0", "vram": 83},
                             "Q8_0": {"kv": "q8_0", "vram": 96}},
             "note": "YaRN-extended; needs the GPU to itself"},
            {"value": 1048576, "label": "1M (YaRN, q8 KV)", "yarn_orig": 262144,
             "kv_default": "q8_0", "per_variant": {"Q8_0": {"vram": 96}},
             "note": "YaRN-extended; needs the GPU to itself"}],
        thinking={"toggleable": False}),
    "qwen35-122b": _fam(
        "Qwen3.5 122B-A10B", "brain", "#22d3ee",
        ["text-to-text", "text-to-code", "agents"],
        base_dir="/mnt/raid1_nvme/models/qwen35-122b", scan_engine="llama.cpp",
        engines={"llama.cpp": _llama()},
        ctx_options=[{"value": 262144, "label": "256K (native)", "kv_default": "f16"}]),
    "minimax-m25": _fam(
        "MiniMax M2.5", "rocket", "#06b6d4",
        ["text-to-text", "text-to-code", "agents"],
        base_dir="/mnt/raid1_nvme/models/minimax-m2.5", scan_engine="llama.cpp",
        engines={"llama.cpp": _llama()},
        ctx_options=[{"value": 65536, "label": "64K (native)", "kv_default": "f16"}]),
    "qwen36-27b-fable-fusion": _fam(
        "Fable Fusion 711", "brain-circuit", "#a855f7",
        ["text-to-text", "vision", "heretic", "mtp"],
        base_dir="/mnt/raid1_nvme/models/qwen36-27b-fable-fusion", scan_engine="llama.cpp",
        engines={"llama.cpp": _llama()},
        ctx_options=[
            {"value": 262144, "label": "256K (native)", "kv_default": "f16",
             "per_variant": {"Q8_0": {"vram": 32}, "Q6_K": {"vram": 24}}},
            {"value": 1048576, "label": "1M (YaRN, q8 KV)", "yarn_orig": 262144,
             "kv_default": "q8_0", "per_variant": {"Q8_0": {"vram": 96}},
             "note": "YaRN-extended; needs most of VRAM"}]),
    "qwen36-27b-fable-amd": _fam(
        "Fable Fusion (AMD)", "cpu", "#f97316",
        ["text-to-text", "vision", "iq4_xs", "amd"],
        base_dir="/mnt/raid1_nvme/models/qwen36-27b-fable-amd", scan_engine="llama.cpp",
        engines={"llama.cpp": _llama()},
        ctx_options=[{"value": 262144, "label": "256K (native)", "kv_default": "f16",
                      "vram": 18}],
        reasoning={"supported": False}, thinking={"toggleable": False}),
    "ornith-35b": _fam(
        "Ornith 1.0 35B", "brain", "#f59e0b", ["llm", "gguf"],
        base_dir="/mnt/raid1_nvme/models/Ornith-1.0-35B", scan_engine="llama.cpp",
        engines={"llama.cpp": _llama()}, ctx_options=[]),
    "laguna-s21": _fam(
        "Laguna S-2.1", "brain", "#7c3aed", ["llm", "gguf", "1m-context"],
        base_dir="/mnt/raid1_nvme/models/Laguna-S-2.1", scan_engine="llama.cpp",
        engines={"llama.cpp": _llama()},
        ctx_options=[{"value": 1048576, "label": "1M (native)", "kv_default": "f16",
                      "per_variant": {"Q8_0": {"vram": 96}, "Q4_K_M": {"vram": 96}}}]),
    "qwen38-flash-next": _fam(
        "Qwen3.8-Flash-Next 125B", "brain-circuit", "#06b6d4",
        ["text-to-text", "vision", "moe", "125b", "qwen4exp"],
        base_dir="/mnt/raid1_nvme/models/qwen38-flash-next", scan_engine="llama.cpp",
        # 125B MoE multimodal (Qwen4Exp arch) — only the PR 27742 build loads it;
        # vram from the user's builtin entries (88G/104G on disk + 262K ctx overhead)
        engines={"llama.cpp": _llama(PR27742_LLAMA,
                                     "/mnt/raid1_nvme/models/llama.cpp-pr27742")},
        ctx_options=[{"value": 262144, "label": "262K (native)", "kv_default": "f16",
                      "per_variant": {"IQ4_XS": {"vram": 94}, "Q4_K_XL": {"vram": 111}}}]),
    "qwen36-27b": _fam(
        "Qwen3.6 27B", "zap", "#f59e0b", ["text-to-text", "text-to-code"],
        engines={
            "vllm": _vllm(VLLM_LEGACY, "/mnt/raid1_nvme/vllm-servers",
                          extra_flags=["--trust-remote-code"]),
            "vllm-nvfp4": _vllm(NVFP4_36, "/mnt/raid1_nvme/models/Qwen3.6-27B-NVFP4",
                                extra_flags=["--trust-remote-code"], env=dict(FLASH_ENV)),
        },
        sources=[
            {"id": "bf16", "kind": "dir", "path": "/mnt/raid1_nvme/models/qwen36-27b",
             "quant": "BF16", "engine": "vllm",
             "ctx_options": [{"value": 131072, "label": "131K (native)",
                              "kv_default": "f16"}]},
            {"id": "nvfp4", "kind": "hf", "path": "unsloth/Qwen3.6-27B-NVFP4",
             "quant": "NVFP4", "weights_gb": 16, "engine": "vllm-nvfp4",
             "ctx_options": [{"value": 262144, "label": "256K (native)",
                              "kv_default": "f16", "vram": 20}]},
        ]),
    "qwen36-35b": _fam(
        "Qwen3.6 35B-A3B", "zap", "#38bdf8", ["text-to-text", "text-to-code"],
        engines={"vllm": _vllm(VLLM_LEGACY, "/mnt/raid1_nvme/vllm-servers",
                               extra_flags=["--trust-remote-code", "--enforce-eager"])},
        sources=[
            {"id": "bf16", "kind": "dir", "path": "/mnt/raid1_nvme/models/qwen36-35b",
             "quant": "BF16", "engine": "vllm",
             "ctx_options": [{"value": 131072, "label": "131K (native)",
                              "kv_default": "f16"}]},
        ]),
    "qwen25-72b": _fam("Qwen 2.5 72B", "brain", "#818cf8", ["text-to-text"],
                       engines={"vllm": _vllm(VLLM_LEGACY, "/mnt/raid1_nvme/vllm-servers")},
                       sources=[{"id": "hf", "kind": "hf",
                                 "path": "Qwen/Qwen2.5-72B-Instruct", "quant": "BF16",
                                 "engine": "vllm",
                                 "ctx_options": [{"value": 32768, "label": "32K",
                                                  "kv_default": "f16"}]}]),
    "qwen25-coder-32b": _fam("Qwen 2.5 Coder 32B", "code", "#a78bfa", ["text-to-code"],
                             engines={"vllm": _vllm(VLLM_LEGACY,
                                                    "/mnt/raid1_nvme/vllm-servers")},
                             sources=[{"id": "hf", "kind": "hf",
                                       "path": "Qwen/Qwen2.5-Coder-32B-Instruct",
                                       "quant": "BF16", "engine": "vllm",
                                       "ctx_options": [{"value": 32768, "label": "32K",
                                                        "kv_default": "f16"}]}]),
    "llama33-70b": _fam("Llama 3.3 70B", "cpu", "#c084fc", ["text-to-text"],
                        engines={"vllm": _vllm(VLLM_LEGACY,
                                               "/mnt/raid1_nvme/vllm-servers")},
                        sources=[{"id": "hf", "kind": "hf",
                                  "path": "meta-llama/Llama-3.3-70B-Instruct",
                                  "quant": "BF16", "engine": "vllm",
                                  "ctx_options": [{"value": 32768, "label": "32K",
                                                   "kv_default": "f16"}]}]),
    "deepseek-r1-32b": _fam("DeepSeek R1 32B", "zap", "#e879f9",
                            ["text-to-text", "reasoning"],
                            engines={"vllm": _vllm(VLLM_LEGACY,
                                                   "/mnt/raid1_nvme/vllm-servers")},
                            sources=[{"id": "hf", "kind": "hf",
                                      "path": "deepseek-ai/DeepSeek-R1-Distill-Qwen-32B",
                                      "quant": "BF16", "engine": "vllm",
                                      "ctx_options": [{"value": 65536, "label": "64K",
                                                       "kv_default": "f16"}]}]),
    "deepseek-r1-70b": _fam("DeepSeek R1 70B", "flame", "#f472b6",
                            ["text-to-text", "reasoning"],
                            engines={"vllm": _vllm(VLLM_LEGACY,
                                                   "/mnt/raid1_nvme/vllm-servers")},
                            sources=[{"id": "hf", "kind": "hf",
                                      "path": "deepseek-ai/DeepSeek-R1-Distill-Llama-70B",
                                      "quant": "BF16", "engine": "vllm",
                                      "ctx_options": [{"value": 32768, "label": "32K",
                                                       "kv_default": "f16"}]}]),
    "qwen35-27b-opus-reasoning": _fam("Qwen3.5 27B Opus Reasoning", "brain-circuit",
                                      "#a855f7", ["text-to-text", "reasoning"],
                                      engines={"vllm": _vllm(
                                          VLLM_LEGACY, "/mnt/raid1_nvme/vllm-servers")},
                                      sources=[{"id": "hf", "kind": "hf",
                                                "path": "Jackrong/Qwen3.5-27B-Claude-4.6-Opus-Reasoning-Distilled",
                                                "quant": "BF16", "engine": "vllm",
                                                "ctx_options": [
                                                    {"value": 32768, "label": "32K",
                                                     "kv_default": "f16"}]}]),
}
```

- [ ] **Step 4: Implement scan helpers (same section, after the dict)**

```python
QUANT_TOKENS = ["Q8_K_XL", "Q8_K_P", "Q6_K_XL", "Q6_K_P", "IQ3_KS", "IQ4_XS", "NVFP4",
                "Q8_0", "Q6_K", "Q5_K_M", "Q4_K_XL", "Q4_K_M", "Q4_0", "BF16", "FP16"]


def parse_quant(filename: str) -> str:
    up = filename.upper()
    for tok in QUANT_TOKENS:
        if tok in up:
            return tok
    return "GGUF"


def _scan_gguf_variants_in(root: str, engine_name: str, exclude: list) -> list:
    """Group *.gguf files under root by quant token; shards → one variant."""
    groups: dict = {}
    for dirpath, _dirnames, filenames in os.walk(root):
        rel = os.path.relpath(dirpath, root).replace("\\", "/")
        if any(ex.rstrip("/") + "/" in rel + "/" for ex in exclude):
            continue
        for fn in filenames:
            if fn.lower().endswith(".gguf"):
                groups.setdefault((rel, parse_quant(fn)), []).append(
                    os.path.join(dirpath, fn))
    out = []
    for (_rel, q), paths in groups.items():
        paths.sort()
        first = next((p for p in paths if "-00001-of-" in os.path.basename(p)), paths[0])
        size = sum(os.path.getsize(p) for p in paths)
        gb = size / 1e9
        out.append({
            "id": f"gguf_{q.lower()}", "quant": q, "path": first,
            "weights_gb": round(size / 1e9, 1), "engine": engine_name, "kind": "gguf",
            "available": True,
            "label": f"{q} · ~{gb:.0f} GB" + (" · shards" if len(paths) > 1 else ""),
            "ctx_options": None,
        })
    return sorted(out, key=lambda v: v["quant"])


def scan_gguf_variants(fid: str) -> list:
    fam = MODEL_FAMILIES[fid]
    eng_key = fam.get("scan_engine")
    if not eng_key or not fam.get("base_dir"):
        return []
    return _scan_gguf_variants_in(fam["base_dir"], eng_key, fam.get("scan_exclude", []))


def scan_sources(fid: str) -> list:
    out = []
    for s in MODEL_FAMILIES[fid].get("sources", []):
        v = dict(s)
        if s["kind"] == "dir":
            try:
                sts = [f for f in os.listdir(s["path"]) if f.endswith(".safetensors")]
            except OSError:
                sts = []
            v["available"] = bool(sts)
            if not v.get("weights_gb") and sts:
                v["weights_gb"] = round(
                    sum(os.path.getsize(os.path.join(s["path"], f)) for f in sts) / 1e9, 1)
            gb = v.get("weights_gb")
            v["label"] = (f'{s["quant"]} · ~{gb:.0f} GB' if gb else f'{s["quant"]} · vLLM dir')
        else:
            v["available"] = True
            gb = s.get("weights_gb")
            v["label"] = f'{s["quant"]} · HF' + (f" · ~{gb:.0f} GB" if gb else "")
        v["ctx_options"] = s.get("ctx_options")
        out.append(v)
    return out


def family_variants(fid: str) -> list:
    return scan_gguf_variants(fid) + scan_sources(fid)


def scan_templates(fid: str) -> list:
    fam = MODEL_FAMILIES[fid]
    found: list = []
    for tdir in fam.get("templates", []):
        try:
            for n in sorted(os.listdir(os.path.join(fam["base_dir"], tdir))):
                if n.endswith(".jinja") and "broken" not in n and n not in found:
                    found.append(n)
        except OSError:
            continue
    return found
```

- [ ] **Step 5: Run — expect PASS**

Run: `cd /mnt/raid1_nvme/JanusPro7b && dashboard-venv/bin/python -m pytest tests/ -v`
If a real-family path test fails because weights genuinely moved, fix the family data — not the
test.

- [ ] **Step 6: Commit**

```bash
git -C /mnt/raid1_nvme/JanusPro7b add -f dashboard.py tests/test_builder_scan.py
git -C /mnt/raid1_nvme/JanusPro7b commit -m "Add MODEL_FAMILIES metadata + disk scan for builder variants"
git -C /mnt/raid1_nvme/JanusPro7b push
```

### Task 3: `resolve_advanced` (validation + emit rules)

**Files:**
- Modify: `dashboard.py` — builder section, after the Task 2 scan helpers
- Create: `tests/test_builder_resolve.py`

**Interfaces:**
- Consumes: `MODEL_FAMILIES`, `scan_templates` (Task 2).
- Produces:
  - `SAMPLING_DEFAULTS: dict[str, dict[str, float]]` — per-engine UI placeholders.
  - `_ctx_for_variant(fam: dict, variant: dict) -> list` — source's own `ctx_options` if present, else the family list.
  - `_resolve_template_path(fid: str, template_val) -> str | None` — absolute `.jinja` path, or `None` for `"builtin"`.
  - `resolve_advanced(fid: str, variant: dict, advanced: dict) -> tuple[dict, list, list]` —
    `(resolved, errors, warnings)`. Error items are `{"field", "message"}`; non-empty `errors`
    means the caller returns 400 and must not use `resolved` (it is `{}` whenever `ctx` itself is
    invalid). `advanced` keys: `ctx` (int, required), `temp`, `top_p`, `repeat_penalty` (float|null),
    `reasoning_effort` (level | `"off"` | null), `enable_thinking` (bool|null), `template`
    (basename | `"builtin"` | null), `kv_cache` (str|null). Resolved keys when valid:
    `ctx:int, ctx_label:str, custom_ctx:bool, yarn_orig:int|None, kv:str|None (None for f16),
    template_path:str|None, llama_kwargs_ok:bool, thinking:bool|None, reasoning:str|None,
    temp/top_p/repeat_penalty:float|None, vram:int|None`.

Emit rules (spec §3, encoded here and nowhere else):
- `reasoning`: support **known True** → always emit (user value else family default; `"off"` → `None`);
  **known False** → hidden, any set value → error; **unknown** → emit only when the user picks a level
  (never by default) + yellow warning.
- `thinking`: known `toggleable` → user value else family default; hidden (`False`) → set value is an
  error; unknown → only when explicitly set + warning. On llama.cpp it is only ever emitted when a
  Jinja template is in effect (file or engine `--jinja`), else error.
- sampling flags: emitted only for non-null (user-changed) values, range-checked.
- `kv_cache`: llama.cpp variants only; defaults from the ctx option's `kv_default` (with per-quant
  override); `f16` resolves to `None` (no flag).

- [ ] **Step 1: Write failing tests**

```python
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
```

- [ ] **Step 2: Run to verify they fail**

Run: `cd /mnt/raid1_nvme/JanusPro7b && dashboard-venv/bin/python -m pytest tests/test_builder_resolve.py -v`
Expected: FAIL `AttributeError: module 'dashboard' has no attribute 'resolve_advanced'`.

- [ ] **Step 3: Implement (same builder section, after the scan helpers)**

```python
SAMPLING_DEFAULTS = {
    "llama.cpp": {"temp": 0.8, "top_p": 0.95, "repeat_penalty": 1.10},
    "vllm": {"temp": 1.0, "top_p": 1.0, "repeat_penalty": 1.0},
    "vllm-nvfp4": {"temp": 1.0, "top_p": 1.0, "repeat_penalty": 1.0},
}


def _ctx_for_variant(fam: dict, variant: dict) -> list:
    return variant.get("ctx_options") or fam.get("ctx_options", [])


def _resolve_template_path(fid: str, template_val) -> str:
    """Absolute .jinja path for the resolved template, or None ('builtin')."""
    fam = MODEL_FAMILIES[fid]
    val = template_val if template_val else fam.get("default_template", "builtin")
    if val == "builtin":
        return None
    cand = os.path.join(fam["base_dir"], val)
    if os.path.isfile(cand):
        return cand
    for tdir in fam.get("templates", []):
        cand = os.path.join(fam["base_dir"], tdir, val)
        if os.path.isfile(cand):
            return cand
    return None


def resolve_advanced(fid: str, variant: dict, advanced: dict) -> tuple:
    fam = MODEL_FAMILIES[fid]
    errors: list = []
    warnings: list = []
    adv = advanced or {}

    ctx = adv.get("ctx")
    if isinstance(ctx, bool) or not isinstance(ctx, int) or not (4096 <= ctx <= 4194304):
        errors.append({"field": "ctx",
                       "message": "ctx must be an integer between 4096 and 4194304"})
        return {}, errors, warnings

    ctxopt = next((c for c in _ctx_for_variant(fam, variant) if c["value"] == ctx), None)
    custom_ctx = ctxopt is None
    if custom_ctx:
        ctxopt = {"value": ctx, "label": f"{ctx // 1024}K", "kv_default": "f16"}
        warnings.append(f"Context {ctx} is not a verified length for {fam['name']}")

    per = ctxopt.get("per_variant", {}).get(variant["quant"], {})
    is_llama = variant["engine"] == "llama.cpp"
    engine = fam["engines"].get(variant["engine"], {})

    # KV cache (llama.cpp only)
    kv = adv.get("kv_cache") or per.get("kv") or ctxopt.get("kv_default", "f16")
    if kv != "f16" and not is_llama:
        errors.append({"field": "kv_cache",
                       "message": "KV cache type only applies to llama.cpp models"})
    if kv not in fam.get("kv_cache", ["f16"]):
        errors.append({"field": "kv_cache",
                       "message": "kv_cache must be one of " + ", ".join(fam.get("kv_cache", []))})

    # Chat template
    tmpl = adv.get("template")
    if tmpl not in (None, "builtin") and tmpl != fam.get("default_template") \
            and tmpl not in scan_templates(fid):
        errors.append({"field": "template", "message": f"unknown template '{tmpl}'"})
        tmpl = None
    tmpl_path = _resolve_template_path(fid, tmpl)
    llama_kwargs_ok = is_llama and (tmpl_path is not None or engine.get("jinja", False))

    # Reasoning effort
    rsup = fam.get("reasoning", {}).get("supported", "unknown")
    rev = adv.get("reasoning_effort")
    reasoning = None
    if rsup is False:
        if rev not in (None, "off"):
            errors.append({"field": "reasoning_effort",
                           "message": "this model has no configurable reasoning effort"})
    elif rsup is True:
        levels = fam["reasoning"].get("levels", ["low", "medium", "high"])
        if rev is None:
            reasoning = fam["reasoning"].get("default")
        elif rev == "off":
            reasoning = None
        elif rev in levels:
            reasoning = rev
        else:
            errors.append({"field": "reasoning_effort",
                           "message": "level must be one of " + ", ".join(levels) + " or 'off'"})
    else:  # unknown
        if rev in (None, "off"):
            reasoning = None
        elif rev in ("low", "medium", "high"):
            reasoning = rev
            warnings.append("Reasoning level not verified for this model")
        else:
            errors.append({"field": "reasoning_effort",
                           "message": "level must be low, medium, high or 'off'"})

    # Thinking toggle
    tcap = fam.get("thinking", {}).get("toggleable", "unknown")
    tev = adv.get("enable_thinking")
    thinking = None
    if tcap is False:
        if tev is not None:
            errors.append({"field": "enable_thinking",
                           "message": "this model has no thinking toggle"})
    elif tcap is True:
        thinking = bool(tev) if tev is not None else bool(fam["thinking"].get("default", False))
    elif tev is not None:
        thinking = bool(tev)
        warnings.append("Thinking toggle not verified for this model")
    if is_llama and thinking is not None and not llama_kwargs_ok:
        errors.append({"field": "enable_thinking",
                       "message": "thinking needs a Jinja chat template"})
        thinking = None

    # Sampling (emitted only when the user sets a value)
    def _num(key: str, lo: float, hi: float, lo_excl: bool = False):
        v = adv.get(key)
        if v is None:
            return None
        if isinstance(v, bool) or not isinstance(v, (int, float)):
            errors.append({"field": key, "message": f"{key} must be a number"})
            return None
        v = float(v)
        if (v <= lo if lo_excl else v < lo) or v > hi:
            errors.append({"field": key,
                           "message": f"{key} must be in {'(' if lo_excl else '['}{lo}..{hi}]"})
            return None
        return v

    temp = _num("temp", 0.0, 2.0)
    top_p = _num("top_p", 0.0, 1.0, lo_excl=True)
    repeat = _num("repeat_penalty", 0.0, 2.0, lo_excl=True)

    resolved = {
        "ctx": ctx,
        "ctx_label": ctxopt.get("label") or f"{ctx // 1024}K",
        "custom_ctx": custom_ctx,
        "yarn_orig": ctxopt.get("yarn_orig"),
        "kv": None if kv == "f16" else kv,
        "template_path": tmpl_path,
        "llama_kwargs_ok": llama_kwargs_ok,
        "thinking": thinking,
        "reasoning": reasoning,
        "temp": temp, "top_p": top_p, "repeat_penalty": repeat,
        "vram": per.get("vram", ctxopt.get("vram")),
    }
    return resolved, errors, warnings
```

- [ ] **Step 4: Run — expect PASS**

Run: `cd /mnt/raid1_nvme/JanusPro7b && dashboard-venv/bin/python -m pytest tests/ -v`

- [ ] **Step 5: Commit**

```bash
git -C /mnt/raid1_nvme/JanusPro7b add -f dashboard.py tests/test_builder_resolve.py
git -C /mnt/raid1_nvme/JanusPro7b commit -m "Add resolve_advanced: builder validation + emit rules"
git -C /mnt/raid1_nvme/JanusPro7b push
```

---

### Task 4: Command + custom-entry builders

**Files:**
- Modify: `dashboard.py` — builder section, after `resolve_advanced`
- Create: `tests/test_builder_cmd.py`

**Interfaces:**
- Consumes: `MODEL_FAMILIES`, `resolve_advanced`, the binary constants (Task 2), `QWEN38_LLAMA`.
- Produces:
  - `make_alias(fid: str, variant: dict, ctx: int) -> str` — sanitized `fid_variantid_ctxK`.
  - `build_launch_cmd(fid: str, variant: dict, resolved: dict, port: int, alias: str) -> tuple[list, dict]`
    — `(cmd, env)`. llama.cpp env is always `{}`; vLLM env = `PYTORCH_ENV` + engine `env`.
  - `build_custom_entry(fid: str, variant: dict, resolved: dict, body: dict, port: int) -> tuple[str, dict]`
    — `(entry_id, entry)` with `entry["id"] == entry_id`; entry shape = spec §4 example plus
    `quant`; `opencode` ref = `{"provider": f"{alias}-{port}", "model_id": alias}`.

Command templates (spec §6, order is normative — tests assert it byte-for-byte):
- llama.cpp: `[bin, --model P, --alias A, --ctx-size N, -ngl 99, -b 2048, -ub 2048,
  --host 0.0.0.0, --port P, --threads 16]` then conditionally, in order: yarn
  (`--rope-scaling yarn --yarn-orig-ctx N`), kv (`-ctk K -ctv K`), template
  (`--chat-template-file F` or `--jinja` when engine jinja and no file), spec draft (family
  `spec_draft` AND engine bin is `QWEN38_LLAMA`: `--spec-draft-model ABS --spec-type draft-mtp
  --spec-draft-ngl 99`), thinking (`--chat-template-kwargs '{"enable_thinking": ...}'`),
  reasoning (`--reasoning-effort L`), user sampling (`--temp/--top-p/--repeat-penalty`, `%g` format).
- vLLM: `[bin, serve, PATH, --served-model-name A, --host 0.0.0.0, --port P,
  --max-model-len N, --gpu-memory-utilization 0.90]` then: engine `dtype`, engine `extra_flags`,
  `--chat-template F` (file only), `--default-chat-template-kwargs '<json>'` with
  `enable_thinking`/`reasoning_effort` keys (omit key and flag when nothing qualifies).

- [ ] **Step 1: Write failing tests**

```python
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
    assert cmd[i + 4:i + 7] == ["-ctk", "q8_0", "-ctv", "q8_0"]
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
```

- [ ] **Step 2: Run to verify they fail**

Run: `cd /mnt/raid1_nvme/JanusPro7b && dashboard-venv/bin/python -m pytest tests/test_builder_cmd.py -v`
Expected: FAIL `AttributeError: module 'dashboard' has no attribute 'make_alias'`.

- [ ] **Step 3: Implement (same builder section, after `resolve_advanced`)**

```python
def make_alias(fid: str, variant: dict, ctx: int) -> str:
    return re.sub(r"[^a-z0-9]+", "_", f"{fid}_{variant['id']}_{ctx // 1024}k").strip("_")


def build_launch_cmd(fid: str, variant: dict, resolved: dict, port: int, alias: str) -> tuple:
    fam = MODEL_FAMILIES[fid]
    engine = fam["engines"][variant["engine"]]
    if variant["engine"] == "llama.cpp":
        cmd = [engine["bin"], "--model", variant["path"], "--alias", alias,
               "--ctx-size", str(resolved["ctx"]),
               "-ngl", "99", "-b", "2048", "-ub", "2048",
               "--host", "0.0.0.0", "--port", str(port), "--threads", "16"]
        if resolved.get("yarn_orig"):
            cmd += ["--rope-scaling", "yarn", "--yarn-orig-ctx", str(resolved["yarn_orig"])]
        if resolved.get("kv"):
            cmd += ["-ctk", resolved["kv"], "-ctv", resolved["kv"]]
        if resolved.get("template_path"):
            cmd += ["--chat-template-file", resolved["template_path"]]
        elif engine.get("jinja"):
            cmd.append("--jinja")
        if fam.get("spec_draft") and engine["bin"] == QWEN38_LLAMA:
            cmd += ["--spec-draft-model",
                    os.path.join(fam["base_dir"], fam["spec_draft"]),
                    "--spec-type", "draft-mtp", "--spec-draft-ngl", "99"]
        if resolved.get("thinking") is not None:
            cmd += ["--chat-template-kwargs",
                    json.dumps({"enable_thinking": bool(resolved["thinking"])})]
        if resolved.get("reasoning"):
            cmd += ["--reasoning-effort", resolved["reasoning"]]
        if resolved.get("temp") is not None:
            cmd += ["--temp", f"{resolved['temp']:g}"]
        if resolved.get("top_p") is not None:
            cmd += ["--top-p", f"{resolved['top_p']:g}"]
        if resolved.get("repeat_penalty") is not None:
            cmd += ["--repeat-penalty", f"{resolved['repeat_penalty']:g}"]
        return cmd, {}
    cmd = [engine["vllm"], "serve", variant["path"], "--served-model-name", alias,
           "--host", "0.0.0.0", "--port", str(port),
           "--max-model-len", str(resolved["ctx"]),
           "--gpu-memory-utilization", "0.90"]
    if engine.get("dtype"):
        cmd += ["--dtype", engine["dtype"]]
    cmd += list(engine.get("extra_flags") or [])
    if resolved.get("template_path"):
        cmd += ["--chat-template", resolved["template_path"]]
    kwargs = {}
    if resolved.get("thinking") is not None:
        kwargs["enable_thinking"] = bool(resolved["thinking"])
    if resolved.get("reasoning"):
        kwargs["reasoning_effort"] = resolved["reasoning"]
    if kwargs:
        cmd += ["--default-chat-template-kwargs", json.dumps(kwargs)]
    env = {**PYTORCH_ENV, **(engine.get("env") or {})}
    return cmd, env


def build_custom_entry(fid: str, variant: dict, resolved: dict, body: dict,
                       port: int) -> tuple:
    fam = MODEL_FAMILIES[fid]
    alias = make_alias(fid, variant, resolved["ctx"])
    entry_id = "cust_" + alias
    cmd, env = build_launch_cmd(fid, variant, resolved, port, alias)
    entry = {
        "id": entry_id,
        "name": f'{fam["name"]} ({variant["quant"]}, {resolved["ctx"] // 1024}K ctx)',
        "description": (body.get("description") or "").strip()
                       or f'{variant["label"]} \u00b7 {variant["engine"]}',
        "port": port,
        "cmd": cmd,
        "cwd": fam["engines"][variant["engine"]].get("cwd", "/"),
        "env": env,
        "protocol": "http",
        "category": "LLM",
        "icon": fam["icon"],
        "color": fam["color"],
        "tags": (body.get("tags") or []) or list(fam.get("tags", [])),
        "supports_offload": variant["engine"] != "llama.cpp",
        "vram_gb": resolved.get("vram") or variant.get("weights_gb"),
        "quant": variant["quant"],
        "custom": True,
        "custom_ref": {"family": fid, "variant": variant["id"], "ctx": resolved["ctx"]},
        "opencode": {"provider": f"{alias}-{port}", "model_id": alias},
    }
    return entry_id, entry
```

Notes for the implementer:
- `json.dumps` default separators are required (they must match the golden strings
  `'{"enable_thinking": true}'` / `'{"enable_thinking": true, "reasoning_effort": "medium"}'`).
- `f"{v:g}"` formats `0.7` → `"0.7"` and `1.0` → `"1"`; the golden tests rely on that.
- The entry's `cwd` comes from the engine block (e.g. `/mnt/raid1_nvme/vllm-servers` for the
  legacy vLLM families), never `"/"` in practice — the `"/"` is a defensive default only.

- [ ] **Step 4: Run — expect PASS**

Run: `cd /mnt/raid1_nvme/JanusPro7b && dashboard-venv/bin/python -m pytest tests/ -v`

- [ ] **Step 5: Commit**

```bash
git -C /mnt/raid1_nvme/JanusPro7b add -f dashboard.py tests/test_builder_cmd.py
git -C /mnt/raid1_nvme/JanusPro7b commit -m "Add build_launch_cmd + build_custom_entry (llama.cpp/vLLM templates)"
git -C /mnt/raid1_nvme/JanusPro7b push
```

---

### Task 5: `custom_models.json` store + port allocation

**Files:**
- Modify: `dashboard.py` — builder section, after `build_custom_entry`
- Create: `tests/test_builder_store.py`

**Interfaces:**
- Consumes: `CUSTOM_MODELS_FILE`, `MODELS`, `os`, `json`; `is_port_open` (defined LATER in the
  file — referenced at call time only, never at module level).
- Produces:
  - `CUSTOM_IDS: set` — module-level set of custom entry ids (empty at import; conftest restores it).
  - `load_custom_entries(path: str) -> list` — `[]` when missing; corrupt file is quarantined to
    `<path>.corrupt-<YYYYmmdd-HHMMSS>`, logged to stderr, `[]` returned (never raises — boot must survive).
  - `save_custom_entries(path: str, entries: list) -> None` — atomic (`tmp` + `os.replace`).
  - `merge_custom_at_startup() -> None` — loads the file, `MODELS[mid] = entry` for each entry,
    skipping (with a stderr log) ids that collide with built-ins; registers `CUSTOM_IDS`.
    Called ONCE at the end of the builder section (module level, after the function is defined).
  - `alloc_port() -> int` — first port in 8100–8199 with no `MODELS` entry and `is_port_open()`
    false; raises `RuntimeError("no free ports in 8100-8199")` when exhausted.

- [ ] **Step 1: Write failing tests**

```python
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
```

- [ ] **Step 2: Run to verify they fail**

Run: `cd /mnt/raid1_nvme/JanusPro7b && dashboard-venv/bin/python -m pytest tests/test_builder_store.py -v`
Expected: FAIL `AttributeError: module 'dashboard' has no attribute 'load_custom_entries'`.

- [ ] **Step 3: Implement (same builder section, after `build_custom_entry`)**

```python
CUSTOM_IDS: set = set()


def load_custom_entries(path: str) -> list:
    if not os.path.exists(path):
        return []
    try:
        with open(path) as f:
            data = json.load(f)
        assert isinstance(data, list)
        return data
    except Exception as e:
        ts = datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
        try:
            os.replace(path, path + f".corrupt-{ts}")
        except OSError:
            pass
        print(f"[builder] {path} unreadable ({e}); continuing with built-ins only",
              file=sys.stderr)
        return []


def save_custom_entries(path: str, entries: list) -> None:
    tmp = path + ".tmp"
    with open(tmp, "w") as f:
        json.dump(entries, f, indent=2)
    os.replace(tmp, path)


def merge_custom_at_startup() -> None:
    for entry in load_custom_entries(CUSTOM_MODELS_FILE):
        mid = entry.get("id")
        if not mid:
            continue
        if mid in MODELS and mid not in CUSTOM_IDS:
            print(f"[builder] custom {mid} collides with existing entry; skipped",
                  file=sys.stderr)
            continue
        MODELS[mid] = entry
        CUSTOM_IDS.add(mid)


def alloc_port() -> int:
    # is_port_open() is defined further down in dashboard.py — fine, called at runtime
    busy = {m["port"] for m in MODELS.values()}
    for port in range(8100, 8200):
        if port not in busy and not is_port_open(port):
            return port
    raise RuntimeError("no free ports in 8100-8199")


merge_custom_at_startup()
```

- [ ] **Step 4: Run — expect PASS** (including the smoke test, which proves the module-level
  `merge_custom_at_startup()` call does not break import — the scratch file doesn't exist yet,
  so it's a no-op)

Run: `cd /mnt/raid1_nvme/JanusPro7b && dashboard-venv/bin/python -m pytest tests/ -v`

- [ ] **Step 5: Commit**

```bash
git -C /mnt/raid1_nvme/JanusPro7b add -f dashboard.py tests/test_builder_store.py
git -C /mnt/raid1_nvme/JanusPro7b commit -m "Add custom_models.json store (atomic, corrupt-safe) + 8100-8199 port allocation"
git -C /mnt/raid1_nvme/JanusPro7b push
```

---

### Task 6: opencode config write-in

**Files:**
- Modify: `dashboard.py` — builder section, after `alloc_port`/startup merge
- Create: `tests/test_builder_opencode.py`

**Interfaces:**
- Consumes: `OPENCODE_CONFIG` (env path, scratch in tests).
- Produces:
  - `opencode_backup(path: str) -> None` — copies `path` → `<path>.bak-<YYYYmmdd-HHMMSS>`,
    keeps the 10 newest `.bak-*` files, deletes older ones. No-op when `path` missing.
  - `opencode_patch(alias: str, port: int, name: str, ctx: int, options: dict) -> str | None`
    — returns a human warning string or `None`. Behaviour (spec §7):
    * file missing → `"opencode config not found — saved to the dashboard only"` (never created);
    * unreadable → `"opencode config unreadable (<err>) — saved to the dashboard only"`;
    * else: backup, then upsert provider `f"{alias}-{port}"`
      `{"npm": "@ai-sdk/openai-compatible", "name": <pid>, "options": {baseURL:
      "http://localhost:<port>/v1", "apiKey": "local", "setCacheKey": false, "timeout": false},
      "models": {…}}` — first reusing any existing provider whose `options.baseURL` already
      matches that port (update-in-place); model block keyed by `alias`:
      `{"name", "limit": {"context": ctx, "output": 32768}, "options": <options> (key present
      only when options non-empty)}`; atomic write; write failure → warning string.
  - `opencode_unpatch(provider_id: str, alias: str) -> None` — best-effort, all failures silent:
    removes `models[alias]` under `provider_id`; if that leaves the provider with zero models
    AND `provider_id.startswith(alias)`, removes the provider too. Nothing else is ever touched.

- [ ] **Step 1: Write failing tests**

```python
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
    for pid in before:
        assert after["provider"].get(pid) == before[pid], pid


def test_backup_rotation_keeps_10(scratch, real_opencode_copy):
    base = os.environ["OPENCODE_CONFIG_PATH"]
    for i in range(1, 13):
        with open(f"{base}.bak-2026{i:02d}01-000000", "w") as f:
            json.dump({}, f)
    d.opencode_patch("rot", 8100, "R", 4096, {})
    assert len(glob.glob(base + ".bak-*")) == 10
```

- [ ] **Step 2: Run to verify they fail**

Run: `cd /mnt/raid1_nvme/JanusPro7b && dashboard-venv/bin/python -m pytest tests/test_builder_opencode.py -v`
Expected: FAIL `AttributeError: module 'dashboard' has no attribute 'opencode_patch'`.

- [ ] **Step 3: Implement (same builder section, after `merge_custom_at_startup()`)**

```python
def opencode_backup(path: str) -> None:
    if not os.path.exists(path):
        return
    ts = datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
    shutil.copy2(path, path + f".bak-{ts}")
    base, fname = os.path.split(path)
    baks = sorted(f for f in os.listdir(base) if f.startswith(fname + ".bak-"))
    for old in baks[:-10]:
        try:
            os.remove(os.path.join(base, old))
        except OSError:
            pass


def opencode_patch(alias: str, port: int, name: str, ctx: int, options: dict):
    path = OPENCODE_CONFIG
    if not os.path.exists(path):
        return "opencode config not found — saved to the dashboard only"
    try:
        with open(path) as f:
            cfg = json.load(f)
    except Exception as e:
        return f"opencode config unreadable ({e}) — saved to the dashboard only"
    opencode_backup(path)
    providers = cfg.setdefault("provider", {})
    url = f"http://localhost:{port}/v1"
    provider_id = f"{alias}-{port}"
    for pid, p in list(providers.items()):
        if (p.get("options") or {}).get("baseURL") == url:
            provider_id = pid
            break
    provider = providers.get(provider_id) or {
        "npm": "@ai-sdk/openai-compatible",
        "name": provider_id,
        "options": {"baseURL": url, "apiKey": "local", "setCacheKey": False,
                    "timeout": False},
        "models": {},
    }
    block = {"name": name, "limit": {"context": ctx, "output": 32768}}
    if options:
        block["options"] = options
    provider.setdefault("models", {})[alias] = block
    providers[provider_id] = provider
    try:
        tmp = path + ".tmp"
        with open(tmp, "w") as f:
            json.dump(cfg, f, indent=2)
        os.replace(tmp, path)
    except Exception as e:
        return f"opencode config write failed ({e}) — saved to the dashboard only"
    return None


def opencode_unpatch(provider_id: str, alias: str) -> None:
    path = OPENCODE_CONFIG
    if not os.path.exists(path) or not provider_id or not alias:
        return
    try:
        with open(path) as f:
            cfg = json.load(f)
    except Exception:
        return
    providers = cfg.get("provider", {})
    changed = False
    if provider_id in providers:
        models = providers[provider_id].get("models", {})
        if alias in models:
            del models[alias]
            changed = True
        if not models and provider_id.startswith(alias):
            del providers[provider_id]
            changed = True
    if changed:
        opencode_backup(path)
        tmp = path + ".tmp"
        with open(tmp, "w") as f:
            json.dump(cfg, f, indent=2)
        os.replace(tmp, path)
```

- [ ] **Step 4: Run — expect PASS**

Run: `cd /mnt/raid1_nvme/JanusPro7b && dashboard-venv/bin/python -m pytest tests/ -v`

- [ ] **Step 5: Commit**

```bash
git -C /mnt/raid1_nvme/JanusPro7b add -f dashboard.py tests/test_builder_opencode.py
git -C /mnt/raid1_nvme/JanusPro7b commit -m "Add opencode config write-in (backup-rotate, update-in-place, safe unpatch)"
git -C /mnt/raid1_nvme/JanusPro7b push
```

---

### Task 7: API endpoints (`/api/families`, `POST/DELETE /api/custom-model`)

**Files:**
- Modify: `dashboard.py` — three routes inserted after `api_logs` (the line
  `return JSONResponse({"logs": text})`), before `@app.get("/")`
- Create: `tests/test_builder_api.py`

**Interfaces:**
- Consumes: everything from Tasks 2–6; `op_lock`; `api_stop` (reused, must be called WITHOUT
  holding `op_lock` — it acquires the same non-reentrant lock).
- Produces:
  - `_family_payload(fid: str) -> dict` — keys: `id, name, docs_url, icon, color, engines[]`
    (sorted unique engine names of AVAILABLE variants), `variants[]` (`id, label, engine, quant,
    weights_gb, available, ctx_options`, with `"description"` when present), `reasoning,
    thinking, templates[]` (`"builtin"` first — for llama families; same list for vLLM-only
    families, which simply won't list a file unless the family has one), `default_template,
    kv_cache[]` (empty for vLLM-only families), `sampling_defaults` (the full `SAMPLING_DEFAULTS`
    map — the UI picks by the selected variant's engine), `tags`.
  - `GET /api/families` → 200, list in `MODEL_FAMILIES` insertion order.
  - `opencode_options(resolved: dict) -> dict` — `temperature`/`topP`/`repetitionPenalty` for
    non-null resolved values only.
  - `POST /api/custom-model` — body `{family, variant, ctx, description?, tags?, advanced?}`.
    Response 200 `{"entry", "warnings", "opencode_warning"?}`; 400 `{"field", "message"}`;
    500 `{"error"}` (port exhaustion or JSON write failure — nothing is saved on 500).
  - `DELETE /api/custom-model/{model_id}` — 404 for built-ins; stops first via `api_stop`
    (no lock held), then under `op_lock` removes from JSON + `MODELS` + `CUSTOM_IDS`; then
    best-effort `opencode_unpatch` using the stored `opencode` ref. Returns 200 `{"message"}`.

Endpoint order of operations (POST) — the JSON write is the commit point:
validate → resolve → duplicate check + `alloc_port` (under `op_lock`) → build entry →
atomic append + `MODELS.update` (under `op_lock`) → opencode patch (best-effort, outside the lock).

- [ ] **Step 1: Write failing tests**

```python
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
```

- [ ] **Step 2: Run to verify they fail**

Run: `cd /mnt/raid1_nvme/JanusPro7b && dashboard-venv/bin/python -m pytest tests/test_builder_api.py -v`
Expected: 404s / assertion failures (routes don't exist yet).

- [ ] **Step 3: Implement the three routes (insert after `api_logs`, before `@app.get("/")`)**

```python
def _family_payload(fid: str) -> dict:
    fam = MODEL_FAMILIES[fid]
    variants = [{
        "id": v["id"], "label": v["label"], "engine": v["engine"], "quant": v["quant"],
        "weights_gb": v.get("weights_gb"), "available": v.get("available", True),
        "ctx_options": _ctx_for_variant(fam, v),
        **({"description": v["description"]} if v.get("description") else {}),
    } for v in family_variants(fid)]
    return {
        "id": fid,
        "name": fam["name"],
        "docs_url": fam.get("docs_url"),
        "icon": fam["icon"],
        "color": fam["color"],
        "engines": sorted({v["engine"] for v in variants if v["available"]}),
        "variants": variants,
        "reasoning": fam.get("reasoning", {}),
        "thinking": fam.get("thinking", {}),
        "templates": ["builtin"] + scan_templates(fid),
        "default_template": fam.get("default_template", "builtin"),
        "kv_cache": list(fam.get("kv_cache", []))
                    if any(v["engine"] == "llama.cpp" for v in variants) else [],
        "sampling_defaults": SAMPLING_DEFAULTS,
        "tags": fam.get("tags", []),
    }


@app.get("/api/families")
def api_families():
    return JSONResponse([_family_payload(fid) for fid in MODEL_FAMILIES])


def opencode_options(resolved: dict) -> dict:
    opts = {}
    if resolved.get("temp") is not None:
        opts["temperature"] = resolved["temp"]
    if resolved.get("top_p") is not None:
        opts["topP"] = resolved["top_p"]
    if resolved.get("repeat_penalty") is not None:
        opts["repetitionPenalty"] = resolved["repeat_penalty"]
    return opts


@app.post("/api/custom-model")
def api_custom_model_create(body: dict):
    fid = body.get("family")
    if fid not in MODEL_FAMILIES:
        return JSONResponse({"field": "family", "message": "unknown family"}, status_code=400)
    variant = next((v for v in family_variants(fid) if v["id"] == body.get("variant")), None)
    if variant is None:
        return JSONResponse({"field": "variant", "message": "unknown variant"}, status_code=400)
    if not variant.get("available", True):
        return JSONResponse({"field": "variant", "message": "weights missing on disk"},
                            status_code=400)
    resolved, errors, warnings = resolve_advanced(
        fid, variant, {** (body.get("advanced") or {}), "ctx": body.get("ctx")})
    if errors:
        return JSONResponse({"field": errors[0]["field"], "message": errors[0]["message"]},
                            status_code=400)
    entry_id = "cust_" + make_alias(fid, variant, resolved["ctx"])
    with op_lock:
        if entry_id in MODELS:
            return JSONResponse({"field": "variant",
                                 "message": f"already exists as {entry_id}"}, status_code=400)
        try:
            port = alloc_port()
        except RuntimeError as e:
            return JSONResponse({"error": str(e)}, status_code=500)
        entry, _ = build_custom_entry(fid, variant, resolved, body, port)
        entries = load_custom_entries(CUSTOM_MODELS_FILE)
        entries.append(entry)
        try:
            save_custom_entries(CUSTOM_MODELS_FILE, entries)   # commit point
        except Exception as e:
            return JSONResponse({"error": f"custom_models.json write failed: {e}"},
                                status_code=500)
        MODELS[entry_id] = entry
        CUSTOM_IDS.add(entry_id)
    warn = opencode_patch(entry["opencode"]["model_id"], port, entry["name"],
                          resolved["ctx"], opencode_options(resolved))
    resp = {"entry": entry, "warnings": warnings}
    if warn:
        resp["opencode_warning"] = warn
    return JSONResponse(resp)


@app.delete("/api/custom-model/{model_id}")
def api_custom_model_delete(model_id: str):
    model = MODELS.get(model_id)
    if model is None or not model.get("custom"):
        return JSONResponse({"error": "not a custom model"}, status_code=404)
    api_stop(model_id)          # stop first, WITHOUT holding op_lock (api_stop takes it)
    with op_lock:
        MODELS.pop(model_id, None)
        CUSTOM_IDS.discard(model_id)
        entries = load_custom_entries(CUSTOM_MODELS_FILE)
        save_custom_entries(CUSTOM_MODELS_FILE,
                            [e for e in entries if e.get("id") != model_id])
    oc = model.get("opencode")
    if oc:
        opencode_unpatch(oc.get("provider", ""), oc.get("model_id", ""))
    return JSONResponse({"message": "Deleted"})
```

- [ ] **Step 4: Run — expect PASS**

Run: `cd /mnt/raid1_nvme/JanusPro7b && dashboard-venv/bin/python -m pytest tests/ -v`

- [ ] **Step 5: Commit**

```bash
git -C /mnt/raid1_nvme/JanusPro7b add -f dashboard.py tests/test_builder_api.py
git -C /mnt/raid1_nvme/JanusPro7b commit -m "Add builder endpoints: GET /api/families, POST/DELETE /api/custom-model"
git -C /mnt/raid1_nvme/JanusPro7b push
```

---

### Task 8: `/api/status` `custom` flag + vLLM offload-detection fix

**Files:**
- Modify: `dashboard.py`
  - `api_status`: add `"custom": model.get("custom", False),` to the per-model result dict
    (insert after the line `"vram_gb": model.get("vram_gb"),`).
  - New module function `is_vllm_cmd` inserted directly above `@app.post("/api/start/{model_id}")`.
  - Inside `api_start`, replace the line `if cmd[0].endswith("vllm"):` with `if is_vllm_cmd(cmd):`.

**Interfaces:**
- Produces: `is_vllm_cmd(cmd: list) -> bool` — true for CLI form (`cmd[0]` ends with `vllm`)
  AND module form (`cmd[1] == "-m"` and `"vllm" in cmd[2]`).
- WHY: the built-in qwen38 vLLM rows (8010/8017) launch
  `.../venv/bin/python -m vllm.entrypoints.openai.api_server ...`; `cmd[0]` is `python`, so the
  CPU-offload toggle silently no-ops on them today. Custom vLLM rows use the `bin/vllm serve`
  form (Task 4) and already matched — this fixes the pre-existing module-form entries only.

- [ ] **Step 1: Write failing tests**

```python
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
```

- [ ] **Step 2: Run to verify they fail**

Run: `cd /mnt/raid1_nvme/JanusPro7b && dashboard-venv/bin/python -m pytest tests/test_status_start.py -v`
Expected: FAIL `AttributeError: module 'dashboard' has no attribute 'is_vllm_cmd'`.

- [ ] **Step 3: Implement**

Above `@app.post("/api/start/{model_id}")`:

```python
def is_vllm_cmd(cmd: list) -> bool:
    """True for vllm CLI form and module form (python -m vllm...)."""
    if not cmd:
        return False
    return cmd[0].endswith("vllm") or (len(cmd) > 2 and cmd[1] == "-m" and "vllm" in cmd[2])
```

In `api_start`, replace:
```python
            if cmd[0].endswith("vllm"):
```
with:
```python
            if is_vllm_cmd(cmd):
```

In `api_status` per-model dict, after `"vram_gb": model.get("vram_gb"),`:
```python
            "custom": model.get("custom", False),
```

- [ ] **Step 4: Run — expect PASS**

Run: `cd /mnt/raid1_nvme/JanusPro7b && dashboard-venv/bin/python -m pytest tests/ -v`

- [ ] **Step 5: Commit**

```bash
git -C /mnt/raid1_nvme/JanusPro7b add -f dashboard.py tests/test_status_start.py
git -C /mnt/raid1_nvme/JanusPro7b commit -m "Add custom flag to /api/status; offload detection now matches module-form vllm cmds"
git -C /mnt/raid1_nvme/JanusPro7b push
```

---

### Task 9: Builder panel UI + custom-row badge/delete

**Files:**
- Modify: `dashboard.py` (`HTML_PAGE`):
  - CSS appended directly before the closing `</style>`.
  - Builder panel HTML inserted directly before `<div class="table-bar">`.
  - JS appended directly before the bootstrap line `refresh();` (~line 2625).
  - `buildRow`: name-cell edit (exact strings below).
- Create: `tests/test_page.py`

**Interfaces:**
- Consumes: `GET /api/families` (Task 7), existing `currentModels`, `refresh()`, `rowMap`,
  `tbody` (JS globals), `/api/status` (Task 8's `custom` flag).
- Produces: DOM ids `builder, builder-body, builder-chev, b-family, b-variant, b-ctx,
  b-ctx-custom, b-desc, b-tags, b-adv, b-adv-body, b-temp, b-topp, b-repp, b-reason, b-think,
  b-template, b-kv, b-sampling-note, b-warn, b-error, b-docs`; JS functions `toggleBuilder,
  loadFamilies, onFamilyChange, curVariant, onVariantChange, onCtxChange, renderWarn, advToggle,
  saveCustomModel, toast, delCustom`; `buildRow` renders a `custom` badge + `✕` delete button for
  custom rows.

Semantics the JS encodes (server re-validates everything):
- Thinking checkbox: known `toggleable` → always sends the checkbox state; unknown → sends
  `true` when checked, `null` when unchecked (a checkbox cannot express explicit-false for
  unknown-capability models; that stays a delete+recreate away and is an accepted v1 limit);
  hidden (`false`) → always `null`.
- Sampling placeholders come from `sampling_defaults[variant.engine]`; the vLLM note
  ("applied via your opencode client, not the server") shows for vLLM variants.
- KV cache + template selects visible per engine (KV only for llama.cpp variants); template
  sends `null` on `"builtin"`.
- Warnings area is client-side only (custom ctx, YaRN notes, unknown-capability picks).

- [ ] **Step 1: Write failing tests**

```python
# tests/test_page.py
import dashboard as d


def test_page_markers():
    h = d.HTML_PAGE
    for marker in [
        'id="builder"', 'id="builder-body"', 'id="b-family"', 'id="b-variant"',
        'id="b-ctx"', 'id="b-ctx-custom"', 'id="b-desc"', 'id="b-tags"',
        'id="b-adv"', 'id="b-adv-body"', 'id="b-adv-body'", 'id="b-temp"',
        'id="b-topp"', 'id="b-repp"', 'id="b-reason"', 'id="b-think"',
        'id="b-template"', 'id="b-kv"', 'id="b-sampling-note"', 'id="b-warn"',
        'id="b-error"', 'id="b-docs"',
        'function toggleBuilder', 'function loadFamilies', 'function onFamilyChange',
        'function onVariantChange', 'function onCtxChange', 'function renderWarn',
        'function advToggle', 'function saveCustomModel', 'function toast',
        'function delCustom', 'class="custom-badge"', 'class="del-btn"',
        'class="builder-toast"',
    ]:
        assert marker in h, marker
    assert "m.custom ? " in h


def test_log_row_colspan_unchanged():
    assert '<td colspan="11">' in d.HTML_PAGE
```

- [ ] **Step 2: Run to verify they fail**

Run: `cd /mnt/raid1_nvme/JanusPro7b && dashboard-venv/bin/python -m pytest tests/test_page.py -v`
Expected: FAIL (missing markers).

- [ ] **Step 3: Implement — CSS (append before the final `</style>`)**

```css
  .builder { background: #1a1b23; border: 1px solid #27272a; border-radius: 8px; margin: 0 0 10px; overflow: hidden; }
  .builder-head { display: flex; align-items: center; justify-content: space-between; padding: 10px 14px; cursor: pointer; user-select: none; }
  .builder-head:hover { background: #1f2028; }
  .builder-title { font-weight: 700; font-size: 0.9em; }
  .builder-chev { color: #52525b; }
  .builder-body { padding: 4px 14px 14px; }
  .builder-body.collapsed { display: none; }
  .builder-row { display: flex; flex-wrap: wrap; gap: 10px; align-items: center; margin-top: 10px; }
  .builder-row.right { justify-content: flex-end; }
  .builder-row label { display: flex; align-items: center; gap: 6px; color: #a1a1aa; font-size: 0.82em; }
  .builder-row .grow { flex: 1 1 260px; }
  .builder-row .grow input { width: 100%; }
  .builder-row input[type=number], .builder-row select { background: #111218; color: #e4e4e7; border: 1px solid #27272a; border-radius: 6px; padding: 5px 8px; font-size: 0.85em; }
  .builder-row input[type=number] { width: 110px; }
  .builder-adv-head { display: inline-flex; align-items: center; gap: 6px; margin-top: 12px; color: #a1a1aa; font-size: 0.82em; cursor: pointer; }
  .builder-adv { background: #14151b; border: 1px solid #27272a; border-radius: 6px; padding: 0 10px 4px; margin-top: 4px; }
  .builder-adv.collapsed { display: none; }
  .builder-note { color: #eab308; font-size: 0.75em; }
  .builder-warn { margin-top: 10px; }
  .builder-warn div { color: #eab308; font-size: 0.78em; padding: 2px 0; }
  .builder-error { color: #f87171; font-size: 0.78em; margin-top: 8px; min-height: 1em; }
  .builder-save { background: #2563eb; color: #fff; border: none; border-radius: 6px; padding: 8px 14px; font-size: 0.85em; font-weight: 600; cursor: pointer; }
  .builder-save:hover { background: #1d4ed8; }
  .custom-badge { display: inline-block; background: #2563eb22; color: #60a5fa; border-radius: 4px; font-size: 0.7em; padding: 1px 5px; margin-left: 6px; vertical-align: middle; }
  .del-btn { cursor: pointer; color: #52525b; margin-left: 6px; font-size: 0.8em; vertical-align: middle; }
  .del-btn:hover { color: #f87171; }
  .builder-toast { position: fixed; bottom: 20px; right: 20px; background: #14532d; color: #bbf7d0; border: 1px solid #166534; padding: 10px 16px; border-radius: 8px; font-size: 0.85em; z-index: 99; }
```

- [ ] **Step 4: Implement — panel HTML (insert directly before `<div class="table-bar">`)**

```html
<div class="builder" id="builder">
  <div class="builder-head" onclick="toggleBuilder()">
    <span class="builder-title">+ New Model Config</span>
    <span class="builder-chev" id="builder-chev">&#9656;</span>
  </div>
  <div class="builder-body collapsed" id="builder-body">
    <div class="builder-row">
      <label>Model family
        <select id="b-family" onchange="onFamilyChange()"></select>
        <a id="b-docs" target="_blank" rel="noopener" hidden>docs &#8599;</a>
      </label>
      <label>Quant <select id="b-variant" onchange="onVariantChange()"></select></label>
      <label>Context <select id="b-ctx" onchange="onCtxChange()"></select></label>
      <input type="number" id="b-ctx-custom" min="4096" max="4194304" step="1024" placeholder="tokens" hidden>
    </div>
    <div class="builder-row">
      <label class="grow">Description <input type="text" id="b-desc" placeholder="(auto-generated if blank)"></label>
      <label class="grow">Tags <input type="text" id="b-tags" placeholder="comma, separated"></label>
    </div>
    <label class="builder-adv-head"><input type="checkbox" id="b-adv" onchange="advToggle()"> Advanced</label>
    <div class="builder-adv collapsed" id="b-adv-body">
      <div class="builder-row">
        <label>Temp <input type="number" id="b-temp" step="0.05" min="0" max="2" placeholder=""></label>
        <label>Top P <input type="number" id="b-topp" step="0.01" min="0.01" max="1" placeholder=""></label>
        <label>Repeat penalty <input type="number" id="b-repp" step="0.01" min="0.01" max="2" placeholder=""></label>
        <span class="builder-note" id="b-sampling-note" hidden>applied via your opencode client, not the server</span>
      </div>
      <div class="builder-row">
        <label>Reasoning effort <select id="b-reason"></select></label>
        <label>Thinking <input type="checkbox" id="b-think"></label>
        <label>Chat template <select id="b-template"></select></label>
        <label>KV cache <select id="b-kv"></select></label>
      </div>
    </div>
    <div id="b-warn" class="builder-warn"></div>
    <div id="b-error" class="builder-error"></div>
    <div class="builder-row right">
      <button class="builder-save" onclick="saveCustomModel()">Save as custom config</button>
    </div>
  </div>
</div>
```

- [ ] **Step 5: Implement — JS (insert before the `refresh();` bootstrap line)**

```js
// ── Model Config Builder ──
let FAMILY_CACHE = null;
let CUR_FAM = null;

function toggleBuilder() {
  const body = document.getElementById('builder-body');
  const open = body.classList.contains('collapsed');
  body.classList.toggle('collapsed', !open);
  document.getElementById('builder-chev').textContent = open ? '\u25be' : '\u25b8';
  if (open && !FAMILY_CACHE) loadFamilies();
}

async function loadFamilies() {
  const r = await fetch('/api/families');
  FAMILY_CACHE = await r.json();
  document.getElementById('b-family').innerHTML =
    FAMILY_CACHE.map(f => `<option value="${f.id}">${f.name}</option>`).join('');
  onFamilyChange();
}

function onFamilyChange() {
  const f = FAMILY_CACHE.find(x => x.id === document.getElementById('b-family').value);
  CUR_FAM = f;
  const docs = document.getElementById('b-docs');
  if (f.docs_url) { docs.href = f.docs_url; docs.hidden = false; } else { docs.hidden = true; }
  document.getElementById('b-variant').innerHTML = f.variants.map(
    v => `<option value="${v.id}"${v.available ? '' : ' disabled'}>${v.label}${v.available ? '' : ' (weights missing)'}</option>`
  ).join('');
  document.getElementById('b-tags').placeholder = (f.tags || []).join(', ');
  onVariantChange();
}

function curVariant() {
  if (!CUR_FAM) return null;
  return CUR_FAM.variants.find(v => v.id === document.getElementById('b-variant').value) || null;
}

function onVariantChange() {
  const v = curVariant();
  if (!v) return;
  const isLlama = v.engine === 'llama.cpp';
  document.getElementById('b-ctx').innerHTML = (v.ctx_options || []).map(
    c => `<option value="${c.value}">${c.label}</option>`
  ).join('') + '<option value="custom">Custom\u2026</option>';
  const sd = (CUR_FAM.sampling_defaults || {})[v.engine] || {};
  document.getElementById('b-temp').placeholder = (sd.temp != null) ? String(sd.temp) : '';
  document.getElementById('b-topp').placeholder = (sd.top_p != null) ? String(sd.top_p) : '';
  document.getElementById('b-repp').placeholder = (sd.repeat_penalty != null) ? String(sd.repeat_penalty) : '';
  document.getElementById('b-sampling-note').hidden = !String(v.engine).startsWith('vllm');
  const kSel = document.getElementById('b-kv');
  kSel.hidden = !isLlama;
  kSel.innerHTML = (isLlama ? (CUR_FAM.kv_cache || ['f16']) : ['f16'])
    .map(k => `<option value="${k}">${k}</option>`).join('');
  kSel.value = 'f16';
  // template options: the family default (file or 'builtin') first, labelled "default",
  // then every other on-disk .jinja file (basenames)
  const tSel = document.getElementById('b-template');
  const tOpts = [];
  if (CUR_FAM.default_template !== 'builtin') {
    tOpts.push(`<option value="${CUR_FAM.default_template}">default (${CUR_FAM.default_template.split('/').pop()})</option>`);
  }
  tOpts.push(`<option value="builtin">${CUR_FAM.default_template !== 'builtin' ? 'engine template (builtin)' : 'default (engine template)'}</option>`);
  for (const t of CUR_FAM.templates) {
    if (t === 'builtin' || t === CUR_FAM.default_template) continue;
    tOpts.push(`<option value="${t}">${t.split('/').pop()}</option>`);
  }
  tSel.innerHTML = tOpts.join('');
  const rs = CUR_FAM.reasoning || {};
  const rSel = document.getElementById('b-reason');
  if (rs.supported === false) { rSel.hidden = true; rSel.innerHTML = ''; }
  else {
    rSel.hidden = false;
    const levels = rs.supported === true ? (rs.levels || ['low', 'medium', 'high']) : ['low', 'medium', 'high'];
    let opts = '';
    if (rs.supported === true) opts += `<option value="">default (${rs.default || 'engine'})</option>`;
    opts += levels.map(l => `<option value="${l}">${l}</option>`).join('')
      + '<option value="off">off (no flag)</option>';
    rSel.innerHTML = opts;
  }
  const tc = CUR_FAM.thinking || {};
  const th = document.getElementById('b-think');
  th.disabled = tc.toggleable === false;
  th.checked = tc.default === true;
  onCtxChange();
}

function onCtxChange() { renderWarn(); }

function renderWarn() {
  const el = document.getElementById('b-warn');
  const v = curVariant();
  if (!v || !CUR_FAM) { el.innerHTML = ''; return; }
  const warns = [];
  const selVal = document.getElementById('b-ctx').value;
  const custom = (selVal === 'custom');
  document.getElementById('b-ctx-custom').hidden = !custom;
  if (custom) {
    warns.push('Context length is not a verified length for this model.');
  } else {
    const c = (v.ctx_options || []).find(x => String(x.value) === selVal);
    if (c && c.note) warns.push(c.note);
  }
  const rs = CUR_FAM.reasoning || {};
  const rv = document.getElementById('b-reason').value;
  if (rs.supported === 'unknown' && rv && rv !== 'off')
    warns.push('Reasoning level not verified for this model.');
  const tc = CUR_FAM.thinking || {};
  if (tc.toggleable === 'unknown' && document.getElementById('b-think').checked)
    warns.push('Thinking toggle not verified for this model.');
  el.innerHTML = warns.map(w => `<div>\u26a0 ${w}</div>`).join('');
}

function advToggle() {
  document.getElementById('b-adv-body').classList.toggle(
    'collapsed', !document.getElementById('b-adv').checked);
}

async function saveCustomModel() {
  const errEl = document.getElementById('b-error');
  errEl.textContent = '';
  const v = curVariant();
  if (!v || !v.available) { errEl.textContent = 'Pick a model family and an available quant.'; return; }
  const selVal = document.getElementById('b-ctx').value;
  const ctx = (selVal === 'custom')
    ? parseInt(document.getElementById('b-ctx-custom').value, 10)
    : parseInt(selVal, 10);
  if (!ctx || ctx < 4096 || ctx > 4194304) {
    errEl.textContent = 'Enter a valid context length (4096\u20134194304) for "Custom\u2026".';
    return;
  }
  const num = id => {
    const x = document.getElementById(id).value.trim();
    return x === '' ? null : parseFloat(x);
  };
  const tc = CUR_FAM.thinking || {};
  const th = document.getElementById('b-think');
  const rSel = document.getElementById('b-reason');
  const kSel = document.getElementById('b-kv');
  const tSel = document.getElementById('b-template');
  const body = {
    family: CUR_FAM.id,
    variant: v.id,
    ctx,
    description: document.getElementById('b-desc').value.trim(),
    tags: document.getElementById('b-tags').value.split(',').map(s => s.trim()).filter(Boolean),
    advanced: {
      temp: num('b-temp'),
      top_p: num('b-topp'),
      repeat_penalty: num('b-repp'),
      reasoning_effort: (rSel.hidden || rSel.value === '') ? null : rSel.value,
      enable_thinking: tc.toggleable === false ? null
        : (tc.toggleable === true ? th.checked : (th.checked ? true : null)),
      template: (tSel.hidden || tSel.value === 'builtin') ? null : tSel.value,
      kv_cache: (kSel.hidden || kSel.value === 'f16') ? null : kSel.value,
    },
  };
  const r = await fetch('/api/custom-model', {
    method: 'POST', headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify(body),
  });
  const j = await r.json().catch(() => ({}));
  if (r.status !== 200) { errEl.textContent = j.message || JSON.stringify(j); return; }
  document.getElementById('b-desc').value = '';
  toast('Saved \u2014 ' + j.entry.name
    + (j.opencode_warning ? ' \u00b7 \u26a0 ' + j.opencode_warning : ''), !!j.opencode_warning);
  refresh();
}

function toast(msg, warn) {
  const t = document.createElement('div');
  t.className = 'builder-toast';
  if (warn) { t.style.background = '#422006'; t.style.color = '#fde68a'; t.style.borderColor = '#713f12'; }
  t.textContent = msg;
  document.body.appendChild(t);
  setTimeout(() => t.remove(), 3000);
}

async function delCustom(id) {
  const name = (currentModels && currentModels[id]) ? currentModels[id].name : id;
  if (!confirm(`Delete \u2018${name}\u2019? Removes the config and its opencode entry; stops it if running.`)) return;
  const r = await fetch('/api/custom-model/' + id, { method: 'DELETE' });
  const j = await r.json().catch(() => ({}));
  if (r.ok) { toast('Deleted ' + name); refresh(); }
  else { toast('Delete failed: ' + (j.error || r.status), true); }
}
```

- [ ] **Step 6: Implement — `buildRow` name cell**

Replace:
```js
    <td class="td-name"><span class="row-icon" style="background:${m.color}18;color:${m.color}">${initial}</span><a class="name-link" id="link-${id}" target="_blank" rel="noopener">${m.name}</a></td>
```
with:
```js
    <td class="td-name"><span class="row-icon" style="background:${m.color}18;color:${m.color}">${initial}</span><a class="name-link" id="link-${id}" target="_blank" rel="noopener">${m.name}</a>${m.custom ? `<span class="custom-badge">custom</span><span class="del-btn" title="delete custom config" onclick="event.stopPropagation();delCustom('${id}')">&#10005;</span>` : ''}</td>
```

- [ ] **Step 7: Run — expect PASS**

Run: `cd /mnt/raid1_nvme/JanusPro7b && dashboard-venv/bin/python -m pytest tests/ -v`

- [ ] **Step 8: Commit**

```bash
git -C /mnt/raid1_nvme/JanusPro7b add -f dashboard.py tests/test_page.py
git -C /mnt/raid1_nvme/JanusPro7b commit -m "Add builder panel UI + custom badge/delete in model table"
git -C /mnt/raid1_nvme/JanusPro7b push
```

---

### Task 10: Full verification (no VRAM, nothing on :80 restarted)

**Files:** no new code unless a check fails (fix belongs in the task whose contract it
violates, in its own commit); this task validates the whole feature end-to-end.

**Interfaces:**
- Consumes: everything; a throwaway `dashboard` instance bound to **127.0.0.1:8099** with
  scratch env overrides. The running `gpu-dashboard.service` (:80) is NEVER touched.

- [ ] **Step 1: Compile + full test suite**

```bash
cd /mnt/raid1_nvme/JanusPro7b && dashboard-venv/bin/python -m py_compile dashboard.py
cd /mnt/raid1_nvme/JanusPro7b && dashboard-venv/bin/python -m pytest tests/ -v
```
Expected: compiles; all tests PASS.

- [ ] **Step 2: JS syntax check (extract the `<script>` block of `HTML_PAGE`)**

```bash
cd /mnt/raid1_nvme/JanusPro7b && dashboard-venv/bin/python - <<'EOF'
import re
src = open("dashboard.py").read()
html = src.split('HTML_PAGE = r"""')[1].split('"""')[0]
blocks = re.findall(r'<script>(.*?)</script>', html, re.S)
open("/tmp/dash_js.js", "w").write("\n;\n".join(blocks))
print(f"extracted {len(blocks)} script block(s)")
EOF
node --check /tmp/dash_js.js && echo "JS syntax OK"
```

- [ ] **Step 3: Confirm no unverified doc links**

```bash
cd /mnt/raid1_nvme/JanusPro7b && dashboard-venv/bin/python -c "import dashboard; assert all(f.get('docs_url') is None for f in dashboard.MODEL_FAMILIES.values()); print('all docs_url None — no dead links shipped')"
```
(If a `docs_url` is ever added, run `curl -sI <url>` and require HTTP 200 before commit.)

- [ ] **Step 4: Throwaway instance + API exercise (127.0.0.1:8099 only)**

```bash
mkdir -p /tmp/builder-tests/final
cp /root/.config/opencode/config.json /tmp/builder-tests/final/opencode_config.json
cat > /tmp/builder-tests/final/run_final.py <<'EOF'
import os, sys
os.environ["DASHBOARD_CUSTOM_MODELS"] = "/tmp/builder-tests/final/custom_models.json"
os.environ["OPENCODE_CONFIG_PATH"] = "/tmp/builder-tests/final/opencode_config.json"
sys.path.insert(0, "/mnt/raid1_nvme/JanusPro7b")
import dashboard, uvicorn
uvicorn.run(dashboard.app, host="127.0.0.1", port=8099, log_level="warning")
EOF
nohup /mnt/raid1_nvme/JanusPro7b/dashboard-venv/bin/python /tmp/builder-tests/final/run_final.py \
  > /tmp/builder-tests/final/server.log 2>&1 &
echo $! > /tmp/builder-tests/final/server.pid
sleep 3
curl -s http://127.0.0.1:8099/api/status | head -c 120
```

Checks (each with the expected outcome):

1. `curl -s http://127.0.0.1:8099/api/families` → array of 19 families; `qwen38-27b` lists
   `gguf_q8_k_xl` (Q8_K_XL) and `gguf_q6_k_xl` (Q6_K_XL) plus `fp16`/`uncensored`/`nvfp4`;
   `qwen36-27b-fable-amd` lists exactly `IQ4_XS`; `qwen38-flash-next` lists `IQ4_XS` + `Q4_K_XL`.
2. `curl -s -o /dev/null -w '%{http_code}' -X POST http://127.0.0.1:8099/api/custom-model
   -H 'Content-Type: application/json' -d '{"family":"nope","variant":"x","ctx":4096}'` → `400`.
3. Valid save (spec §10 case):
   ```bash
   curl -s -X POST http://127.0.0.1:8099/api/custom-model -H 'Content-Type: application/json' \
     -d '{"family":"qwen38-27b","variant":"gguf_q8_k_xl","ctx":262144,
          "advanced":{"temp":0.6,"enable_thinking":false}}' \
     > /tmp/builder-tests/final/post1.json
   ```
   Expected: `entry.port == 8100`; `entry.cmd` byte-identical to Task 4's golden cmd with
   port `8100` and `--chat-template-kwargs '{"enable_thinking": false}'`; no `topP`/
   `repetitionPenalty` involvement; scratch opencode file gains provider
   `qwen38_27b_gguf_q8_k_xl_256k-8100` with model block `limit.context == 262144` and
   `options == {"temperature": 0.6}` only; a `*.bak-*` file exists; `/api/status` contains the
   id with `"custom": true`.
4. Second save (different variant, e.g. `gguf_q6_k_xl`) → port `8101`. Duplicate of step 3 →
   `400` with `already exists as cust_qwen38_27b_gguf_q8_k_xl_256k`.
5. `DELETE /api/custom-model/cust_qwen38_27b_gguf_q8_k_xl_256k` → row gone from `/api/status`;
   scratch JSON now holds only the q6 entry; the opencode model (and its now-empty provider) is
   gone and built-in providers are byte-identical to before; a fresh re-save of the q8 config →
   port `8100` again (first-free reuse).
6. Corrupt-boot: `kill $(cat /tmp/builder-tests/final/server.pid)`,
   `echo '{bad' > /tmp/builder-tests/final/custom_models.json`, restart the same runner →
   `/api/status` still 200 with **built-ins only** (no `cust_` ids) and the server log contains
   the "unreadable" message.

- [ ] **Step 5: Cleanup**

```bash
kill $(cat /tmp/builder-tests/final/server.pid) 2>/dev/null
rm -rf /tmp/builder-tests/final /tmp/dash_js.js
```
Confirm `curl -s -o /dev/null -w '%{http_code}' http://127.0.0.1:80/api/status` (the real
dashboard) is still `200` — proof nothing on :80 was restarted — and
`nvidia-smi --query-gpu=memory.used,power.draw --format=csv,noheader` shows no model-server
growth. **No model server was ever started during any of this.**

- [ ] **Step 6: Record the one open manual check (non-blocking)**

The opencode binary contains `repetitionPenalty`/`repetition_penalty` strings (verified in
planning), so the builder writes `repetitionPenalty` into the model `options`. Optional manual
follow-up for the user: in a live opencode session pointed at a custom model with a
repeat-penalty set, confirm the outbound request body carries `repetition_penalty`. If it does
not, add the yellow note from spec §7 ("repeat-penalty not forwarded by opencode") to the
builder UI. Record the outcome here by annotating this step in the commit message if acted on.

- [ ] **Step 7: Commit & push (only if Steps 1–6 required fixes)**

```bash
git -C /mnt/raid1_nvme/JanusPro7b status --short
# if clean: done. If fixes were needed, each fix is already committed by its owning task;
# push any unpushed commits:
git -C /mnt/raid1_nvme/JanusPro7b push
```
