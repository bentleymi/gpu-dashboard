# Model Config Builder — Design Spec

Date: 2026-08-26
Status: Approved in chat (approach A + 5-section design), pending spec review
Scope: `dashboard.py` (single-file FastAPI app, `/mnt/raid1_nvme/JanusPro7b`) + `~/.config/opencode/config.json`

## 1. Goal

Let the user build a new launch configuration for any LLM model family on disk from a panel
in the dashboard:

**Family → Quant (only what exists on disk) → Context (only known-safe options) → Description/Tags →
Advanced (temp, top_p, repeat-penalty, reasoning effort, enable_thinking, chat template, KV cache type)**
**→ Save**

Saving creates a **custom model** that appears immediately as a normal row in the existing model
table (start/stop, logs, CPU offload, favorites, search, sort all work unchanged) and, for the user's
main client, an entry in `~/.config/opencode/config.json`.

Steering rules (hard requirements):

1. Options default to the model's own defaults. A blank advanced field means "engine/family default"
   and produces no flag.
2. A choice is only offered if it is known to exist / be supported (e.g. no quant that isn't on disk,
   no `xhigh` reasoning level the model lacks).
3. Where support is genuinely unknown, the option is still allowed but carries a visible yellow
   "not verified for this model" warning.
4. Doc links only where a real, verified URL exists; otherwise no link (never a dead one).

Non-goals: editing an existing custom config (delete + re-create covers it), building configs for
image/video/audio/tool models, launching models during development (zero VRAM use in testing).

## 2. Where this lives in the code

Current structure of `dashboard.py` (2586 lines, single file):

- `MODELS` dict, lines 11–1258: `id → {name, description, port, cmd[], cwd, env, protocol,
  category, icon, color, tags[], supports_offload, optional: systemd_service, vram_gb, path}`.
- `processes`, `log_files`, `op_lock` (threading.Lock) at lines 1263–1265.
- `is_port_open` (1309), `get_model_status` (1317: managed proc → systemd → port check).
- `QUANT_PATTERNS` + `get_quant` (1430–1467): heuristic quant detection from name/description/cmd.
- `/api/status` (1470): the dict the frontend renders one row from per model.
- `/api/start/{id}?offload=` (1499): offload = `cmd[0].endswith("vllm")` → append
  `--cpu-offload-gb 24`; else `CPU_OFFLOAD=1` env. Popen → `logs/{id}.log`, `setsid`.
- `/api/stop/{id}` (1569), `/api/logs/{id}` (1607: last 200 lines of log file, else journalctl).
- `HTML_PAGE` (1638–2583): table rendered from `/api/status` by JS `buildRow(id, m)` (2217) /
  `updateRow` / `renderAll`; columns fixed at 11 (`table-layout: fixed`); favorites in
  `localStorage["fav_models"]`; default sort `fav` desc; 5 s poll (`refresh`).
- `uvicorn.run(app, host="0.0.0.0", port=80)` (2585–2586).

Everything custom-models need already flows through `MODELS` + `/api/status`: merged entries appear
in the table automatically. New surface = one metadata table, one JSON file, three API endpoints,
one builder panel in the HTML, one `custom` flag in the status response.

## 3. Family metadata (`MODEL_FAMILIES` in `dashboard.py`)

One dict entry per builder-eligible LLM family. Capability data is curated (hand-written, small);
weight presence/sizes and GGUF variant lists are **detected from disk at request time**, so deleting
weights removes the options and no metadata can go stale.

Schema:

```python
MODEL_FAMILIES = {
    "qwen38-27b": {
        "name": "Qwen3.8 27B",
        "icon": "sparkles",
        "color": "#f97316",
        "docs_url": "<verified HF/docs URL or None>",
        "base_dir": "/mnt/raid1_nvme/Qwen3_8-27B",        # scanned for GGUF variants
        "scan_exclude": ["gguf-q8/mtp"],                  # draft-model dirs are not variants
        "spec_draft": "gguf-q8/mtp/MTP/mtp-Qwen3.8-27B-Q4_0.gguf",  # optional, llama.cpp only
        "engines": {
            # defined only for engines this family actually uses; a variant's effective
            # engine is a sources entry's "engine" override or the GGUF scan engine
            "llama.cpp": {"bin": ".../llama.cpp/build/bin/llama-server",
                          "cwd": "/mnt/raid1_nvme/Qwen3_8-27B",
                          "jinja": False},                # families using --jinja set True
            "vllm": {"vllm": "/mnt/raid1_nvme/Qwen3_8-27B/venv/bin/vllm",
                     "cwd": "/mnt/raid1_nvme/Qwen3_8-27B",
                     "dtype": "half"},                    # optional --dtype
            "vllm-nvfp4": {"vllm": "/mnt/raid1_nvme/models/Qwen3.8-27B-NVFP4/venv/bin/vllm",
                           "cwd": "/mnt/raid1_nvme/models/Qwen3.8-27B-NVFP4",
                           "env": {"FLASHINFER_DISABLE_VERSION_CHECK": "1"}},
        },
        "sources": [                                       # vLLM sources (local dir or HF id)
            {"id": "fp16", "kind": "dir", "path": "/mnt/raid1_nvme/Qwen3_8-27B/fp16",
             "quant": "FP16"},
            {"id": "uncensored", "kind": "dir", "path": "/mnt/raid1_nvme/Qwen3_8-27B/uncensored",
             "quant": "FP16", "weights_gb": 55,
             "description": "orcarouter abliterated finetune"},
            {"id": "nvfp4", "kind": "hf", "path": "unsloth/Qwen3.8-27B-NVFP4",
             "quant": "NVFP4", "weights_gb": 16,
             "engine": "vllm-nvfp4"},                     # engine key override (own venv)
        ],
        # ctx_options: family-level list applies to all GGUF variants of this family;
        # a `sources` entry may OVERRIDE with its own "ctx_options" (same shape) when the
        # variant's safe context differs (e.g. BF16 dir runs 131K, NVFP4 runs 256K).
        "ctx_options": [
            {"value": 262144, "label": "256K (native)", "yarn_orig": None,
             "kv_default": "f16", "vram": 80, "note": None},
            # + {"value": 524288, "label": "512K (YaRN)", "yarn_orig": 262144,
            #    "kv_default": "q8_0", "vram": 84,
            #    "note": "YaRN-extended context; quality may degrade past native"},
        ],
        "reasoning": {"supported": True, "levels": ["low", "medium", "high"],
                      "default": "medium"},               # or {"supported": False}
        "thinking": {"toggleable": True, "default": True},
        "templates": ["templates"],                        # dirs under base_dir scanned for *.jinja
        "default_template": "templates/sharp-chat-template-v22.1.1.jinja",  # or "builtin"
        "kv_cache": ["f16", "q8_0", "q4_0"],              # llama.cpp only
        "tags": ["text-to-text", "text-to-code"],          # pre-filled tag suggestions
    },
    # ...
}
```

A family defines **only the engine blocks it needs** (a GGUF-only family has one `llama.cpp`
entry; the NVFP4 families have only their `vllm-nvfp4` block). A variant's effective engine is
`source.engine` override (for `sources` entries) or the engine of the detected GGUF dir
(`llama.cpp` / `llama_cpp.server`, per family).

Field semantics:

- **Variants** = the union of (a) GGUF variants detected under `base_dir`: all `*.gguf` files
  (excluding paths in `scan_exclude`) grouped by quant token parsed from the filename — one file or
  all shards of one quant = one variant; the cmd uses the `-00001-of-*` shard (or the single file);
  size = summed bytes — and (b) `sources` entries (`kind: dir` only if the dir contains
  `*.safetensors`; `kind: hf` always offered, labelled "HF cache").
  Grouping per-quant (not per-dir) handles dirs that hold several quants (e.g.
  `davidau-qwen3.6-40b/` has `Q6_K` and `HIGH-Q8_0` side by side).
- **`reasoning`**: levels list is the complete set of allowed `reasoning_effort` values.
  `supported: False` → field hidden. Unknown support (metadata says `unknown`) → dropdown shown
  (levels `low/medium/high` + "off") with a "not verified for this model" warning.
- **Emit rule for thinking + reasoning** (both engines, one rule):
  - family support **known** (explicit metadata default): always emit the resolved value
    (user value, else family default) — this reproduces existing entries (8010/8017: thinking true +
    medium; 8086–8098: thinking false; 8011/8016: medium).
  - family support **unknown**: emit only when the user explicitly changes the field (thinking) or
    picks a level (reasoning; "off" = no flag). No unverified flag is ever emitted by default.
  - the reasoning dropdown also offers "off (no flag)" for known families (existing entries like
    8082 omit the flag entirely).
- **`thinking`**: toggle. `toggleable: False` → hidden. The resolved value is emitted only when a
  Jinja chat template is in effect (below): llama.cpp
  `--chat-template-kwargs '{"enable_thinking":…}'`, vLLM the same key inside
  `--default-chat-template-kwargs`.
- **`templates` / `default_template`**: the template dropdown = `default_template` (labelled
  "default") plus each `*.jinja` file actually found (`.broken*` excluded). Rule for the cmd:
  - llama.cpp + resolved template is a file (default or user-picked) →
    `--chat-template-file <path>`; user-picked ≠ default additionally means nothing else changes.
  - llama.cpp + `default_template: "builtin"` → `--jinja` when the family's `jinja` metadata is
    true (coder-next, 35b-unc*, 40b*, 122b, m2.5, fable*); no template flag otherwise.
  - vLLM + resolved template is a file → `--chat-template <path>`; else the weights' embedded
    HF `chat_template` (the qwen36-27b/35b dirs carry one) is used with no flag.
  - **thinking kwargs are emitted only under a Jinja template** (a file, or `--jinja`); if the
    family's resolved template can't take kwargs the field is hidden. In practice every seeded
    family with `thinking.toggleable: true` resolves to a Jinja template, so the rule only matters
    if a template dir is empty for such a family in the future.
- **`ctx_options`**: curated, known-safe entries, **per variant** — a `sources` entry with its own
  `ctx_options` uses those; everything else uses the family-level list. The context dropdown is
  filled for the currently selected variant and offers **"Custom…"** in addition:
  which reveals a numeric input (tokens, 4096…4194304) and a permanent yellow "not verified for
  this model" warning. `yarn_orig` → `--rope-scaling yarn --yarn-orig-ctx N` (llama.cpp) or
  `--max-model-len` only (vLLM, no RoPE flag). `kv_default` → pre-selects the KV-cache-type advanced
  field for that ctx. `vram` (optional): total VRAM estimate copied from the user's existing
  dashboard entries for equivalent launches.
- **`docs_url`**: curated; verified with a live `curl -sI` check during implementation; `None`
  (no link rendered) when the model is a private/custom build with no canonical page.
- **`vram` display rule for custom rows** (the table's VRAM column): curated `vram` total for the
  chosen ctx if present, else the variant's on-disk `weights_gb` (column shows `~N GB` either way;
  it is an estimate). Estimate > 90 GB of the 96 GB card → value rendered red.

### Seeded family table (initial curation)

GGUF = ik build (`/mnt/raid1_nvme/models/ik_llama.cpp/build/bin/llama-server`) unless noted.
`w/` = vram estimate source: `E` = copied from an existing dashboard entry, `D` = measured on disk
at request time, `—` = weights only.

| family id | name | engines / variants (on-disk quants) | ctx options | reasoning | thinking | templates |
|---|---|---|---|---|---|---|
| `qwen38-27b` | Qwen3.8 27B | llama.cpp (own build): gguf-q8 `Q8_K_XL`, gguf-q6 `Q6_K_XL` (+MTP draft); vllm `fp16`, `uncensored`, vllm-nvfp4 `NVFP4` (HF) | 256K native (E per variant: fp16 80 / unc 56 / q8 46 / q6 40 / nvfp4 20); rest via custom+warn | [low,medium,high] def med | yes, def true | sharp + qwen38-safe |
| `qwen3-coder-next` | Qwen3-Coder-Next | GGUF: Q6_K, Q4_K_M (sharded) | 131K native only | hide (uses `--reasoning-tokens none`) | hide | none |
| `qwen36-35b-uncensored` | Qwen3.6 35B Uncensored | GGUF: Q8_K_P, Q6_K_P | 256K; 512K YaRN; 1M YaRN q8 KV (84 E) | unknown → warn | yes, def false | none |
| `davidau-40b` | Qwen3.6-40B Deck Opus | GGUF: Q6_K, HIGH-Q8_0 | 256K; 384K q4 KV (81 E); 512K (83 E q6 / 96 E q8); 1M q8 KV (96 E) | unknown → warn | none supported → hide | none |
| `qwen35-122b` | Qwen3.5 122B-A10B | GGUF: Q5_K_M, Q4_K_M (sharded) | 256K native | unknown → warn | unknown → warn | none |
| `minimax-m25` | MiniMax M2.5 | GGUF: smol-IQ3_KS (sharded) | 64K native (existing uses 64K) | unknown → warn | unknown → warn | none |
| `qwen36-27b-fable-fusion` | Fable Fusion 711 | GGUF: Q8_0 MTP, Q6_K MTP | 256K (32 E); 1M YaRN q8 KV (96 E) | unknown → warn | unknown → warn | none |
| `qwen36-27b-fable-amd` | Fable Fusion (AMD) | GGUF: IQ4_XS MTP | 256K (18 E) | hide | hide | none |
| `ornith-35b` | Ornith 1.0 35B | llama_cpp.server (own venv): Q8_0 (38 E), BF16 (72 E) | unknown → custom only | hide (engine) | hide (engine) | none |
| `laguna-s21` | Laguna S-2.1 | llama_cpp.server (poolside venv): Q4_K_M, Q8_0 (96 E) | 1M native | hide (engine) | hide (engine) | none |
| `qwen36-27b` | Qwen3.6 27B | vllm dir `/mnt/raid1_nvme/models/qwen36-27b` (BF16); vllm-nvfp4 `NVFP4` (HF, 20 E) | BF16: 131K · NVFP4: 256K (per-variant ctx) | unknown → warn | unknown → warn | none |
| `qwen36-35b` | Qwen3.6 35B-A3B | vllm dir `/mnt/raid1_nvme/models/qwen36-35b` (+`--enforce-eager`) | 131K native | unknown → warn | unknown → warn | none |
| `qwen25-72b`, `qwen25-coder-32b`, `llama33-70b`, `deepseek-r1-32b`, `deepseek-r1-70b`, `qwen35-27b-opus-reasoning` | legacy HF-vLLM | `kind: hf` via `/mnt/raid1_sata/vllm-env/bin/vllm`, single variant each | 32K (64K for R1-32B) | unknown → warn | unknown → warn | none |

Excluded from the builder (no clean engine template, or non-LLM): Gradio/chat-script entries
(`qwen35b` 8007, `qwen36_35b_chat` 8014, `qwen30b_abliterated` 8008), `gemma_4_31b`
(transformers CLI), all Image/Video/Audio/Tools.

## 4. Custom config persistence

- File: `/mnt/raid1_nvme/JanusPro7b/custom_models.json` (path overridable by env
  `DASHBOARD_CUSTOM_MODELS` for testing; same env-override pattern for the opencode config path,
  `OPENCODE_CONFIG_PATH`).
- Contents: JSON array; each item = the **complete model entry** plus bookkeeping:

```json
{
  "name": "Qwen3.8 27B (Q8_K_XL, 256K ctx)",
  "description": "...", "port": 8101, "cmd": ["..."], "cwd": "...", "env": {},
  "protocol": "http", "category": "LLM", "icon": "sparkles", "color": "#f97316",
  "tags": ["..."], "supports_offload": false, "vram_gb": 46,
  "custom": true,
  "custom_ref": {"family": "qwen38-27b", "variant": "gguf_q8", "ctx": 262144},
  "opencode": {"provider": "qwen38_27b_gguf_q8_262k-8101", "model_id": "qwen38_27b_gguf_q8_262k"}
}
```

- **Model id / slug**: `cust_` + sanitized `family_variant_ctxK`
  (e.g. `cust_qwen38_27b_gguf_q8_262k`); the LLM `--alias` / vLLM `--served-model-name` is the slug
  without the `cust_` prefix. Must not collide with a `MODELS` key or an existing custom id.
- **Merge at startup**: after `MODELS` is defined, `MODELS.update(loading from JSON)`. Corrupt or
  unreadable file → log to stderr, continue with built-ins only (the dashboard must never fail to
  boot because of this file). Id colliding with a built-in → skip entry, log.
- **Merge at save time**: same update in the running process → the new row appears on the next
  5 s poll (and an immediate `refresh()` after save), no service restart.
- **Writes are atomic** (`tmp` file + `os.replace`) and serialized under `op_lock` (reused).
- Save-time validation (server-side, returns `400 {field, message}` on failure):
  family exists; variant exists **and is present on disk right now**; ctx is a known option or an
  explicit custom int in range; advanced values within type/range; duplicate
  `(family, variant, ctx)` already saved → 400 "already exists as `<id>`".

### Port allocation

First free port in **8100–8199**: not used by any `MODELS` entry (built-in + custom) and
`is_port_open()` false. Range exhausted → 500 "no free ports in 8100–8199" (user can stop a custom
or extend the range in code).

## 5. API

### `GET /api/families`

List of family objects for the builder form: `id, name, docs_url, engines (which engines have ≥1
available variant)`, `variants[]` (`id, label` e.g. "Q8_K_XL · ~30 GB", `engine`, `weights_gb`,
`available`, `ctx_options[]` for this variant per the §3 per-variant rule — "Custom…" is
handled client-side), `reasoning, thinking, templates[]` (id `default` + each `*.jinja` filename
actually found in the family's templates dirs, `*.broken*` excluded), `kv_cache[]` (llama.cpp
families only), `sampling_defaults` (placeholders shown as field hints: llama.cpp `temp 0.8 /
top_p 0.95 / repeat-penalty 1.10`; vLLM `1.0 / 1.0 / 1.0`), `tags` suggestions.

### `POST /api/custom-model`

Body:

```json
{
  "family": "qwen38-27b", "variant": "gguf_q8",
  "ctx": 262144,                      // or custom int
  "description": "",                  // empty → auto-generated
  "tags": ["a", "b"],
  "advanced": {
    "temp": null, "top_p": null, "repeat_penalty": null,
    "reasoning_effort": null, "enable_thinking": null,
    "template": null, "kv_cache": null
  }                                   // null = family/engine default
}
```

Behaviour: validate (§4) → build cmd (§6) → allocate port → assemble entry → **atomic append to
`custom_models.json` (the commit point)** → `MODELS.update` → best-effort opencode patch (§7) →
200 with the stored entry. The opencode patch never blocks the save: on any failure the response
includes `opencode_warning` (the row works regardless).

### `DELETE /api/custom-model/{id}`

404 for built-ins. Under `op_lock`: stop the model if running (same path as `/api/stop`), remove
the entry from JSON + `MODELS`, remove its opencode provider/model block (only if the stored
`opencode` ref exists). Existing log file left in place.

### `/api/status` change (one line)

Add `"custom": model.get("custom", False)` to the per-model dict so `buildRow` can render custom
rows.

## 6. Command generation

Sampling flags are emitted **only when the user changed the value** (non-null). "Family default"
values for thinking/reasoning on vLLM are emitted via `--default-chat-template-kwargs` because that
is how the existing 8010/8017 entries behave (the server needs them to render thinking tokens).

### llama.cpp (`llama-server`)

```
<bin> --model <gguf> --alias <slug> --ctx-size <ctx> -ngl 99 -b 2048 -ub 2048
      --host 0.0.0.0 --port <port> --threads 16
```

Conditional (in this order):
- ctx.yarn_orig → `--rope-scaling yarn --yarn-orig-ctx <orig>`
- kv_cache ≠ f16 → `-ctk <kv> -ctv <kv>`
- chat template per the §3 rule: `--chat-template-file <path>` (resolved template is a file) or
  `--jinja` (builtin + `jinja: true` family: coder-next, 35b-unc*, 40b*, 122b, m2.5, fable*)
- family `spec_draft` present and variant's engine is the qwen3.8 build →
  `--spec-draft-model <draft> --spec-type draft-mtp --spec-draft-ngl 99`
- thinking (resolved value, family has `thinking` metadata):
  `--chat-template-kwargs '{"enable_thinking":<bool>}'`
- reasoning: `--reasoning-effort <level>` (only when a level is chosen; blank = omit)
- user-changed: `--temp <x>`, `--top-p <x>`, `--repeat-penalty <x>`

### `llama_cpp.server` (ornith, laguna)

```
<venv>/bin/python -m llama_cpp.server --model <gguf> --port <port> --host 0.0.0.0
```
+ `--n-ctx <ctx>`, `--temp/--top-p/--repeat_penalty` when changed. No YaRN/KV/template/thinking
options for these (no such flags → fields hidden). `supports_offload: False`.

### vLLM

```
<vllm bin> serve <dir-or-HF-id> --served-model-name <slug>
  --host 0.0.0.0 --port <port> --max-model-len <ctx> --gpu-memory-utilization 0.90
```
Conditional: `--dtype half` (qwen38 family), `--enforce-eager` (35b), family `extra_flags`,
`--chat-template <path>` per the §3 template rule, `--default-chat-template-kwargs '<json>'` with
the `{enable_thinking, reasoning_effort}` keys per the §3 emit rule (omit the key, and the flag
itself, when nothing qualifies).
`supports_offload: True`; **custom vLLM cmds always use the `<venv>/bin/vllm serve` form** so the
existing offload detection in `api_start` (`cmd[0].endswith("vllm")`, dashboard.py:1522) works.
`env` = the engine block's `env` (e.g. `FLASHINFER_DISABLE_VERSION_CHECK` for `vllm-nvfp4`,
matches 8088/8094) plus `PYTORCH_ALLOC_CONF: "expandable_segments:True"` on every vLLM entry
(matches 8010).
`--enable-auto-tool-choice --tool-call-parser` is **not** added by the builder (user can request it
later; keeping the template minimal).

**Related fix** (small, needed for the module-style entries that predate this feature): `api_start`'s
offload detection also matches `python -m vllm.entrypoints…` commands
(`cmd[:2] == [..., "-m"]` and `"vllm" in cmd[1]`). Without it, the CPU-offload toggle on
built-in 8010/8017-style rows silently does nothing.

## 7. opencode config write-in (`~/.config/opencode/config.json`)

Every saved custom config gets an opencode provider+model entry (this is also how llama.cpp customs
become usable in opencode — an approved part of the design).

- **Provider** `id = <slug without the cust_ prefix>-<port>`
  (e.g. `qwen38_27b_gguf_q8_262k-8100`), matching the `opencode.provider` stored on the entry:
  `npm: "@ai-sdk/openai-compatible"`, `name: <provider id>`,
  `options: {baseURL: "http://localhost:<port>/v1", apiKey: "local"}`,
  `setCacheKey: false`, `timeout: false` (mirrors existing entries).
- **Model block** keyed by `<slug without prefix>` under that provider (this is exactly the
  `--alias` / `--served-model-name` the server uses, so opencode and the server agree):
  `name: <entry name>`,
  `limit: {context: <chosen ctx>, output: 32768}`, `options` containing
  `temperature` / `topP` / `repetitionPenalty` **only for values the user set** (non-null).
  `repetitionPenalty` is written only if implementation verification (see §10) shows opencode's
  openai-compatible provider forwards it as `repetition_penalty`; otherwise the builder shows a
  yellow note "repeat-penalty not forwarded by opencode — value stored in the dashboard config
  only" and the field stays editable.
- **Update-in-place**: before creating, look for any existing model block whose
  `baseURL` matches the allocated port; if found, reuse that provider id and replace only the model
  block.
- **Backup**: `config.json` → `config.json.bak-<YYYYmmdd-HHMMSS>` immediately before writing;
  keep the 10 most recent backups, delete older ones. Writes are atomic.
- **Delete**: remove the model block; if the provider then has zero models and its id starts with a
  custom slug, remove the provider too. Built-in provider entries are never touched.

## 8. UI

### Builder panel (above the table bar)

Collapsible (collapsed by default), matching the existing dark palette (`#1a1b23` panel,
`#27272a` borders, Inter font):

```
[+ New Model Config                       ▾]
┌──────────────────────────────────────────────────────────────────────┐
│ Model family ▾   Quant ▾   Context ▾   [131072] (only for "Custom…") │
│ Description [______________]  Tags [____________]                    │
│ ☐ Advanced                                                                    │
│  Temp [ ]  Top P [ ]  Repeat penalty [ ]   (placeholders = engine defaults)  │
│  Reasoning effort ▾ (family levels only)   Thinking (toggle)           │
│  Chat template ▾ (default + found .jinja files)   KV cache ▾ (llama.cpp) │
│  ⚠ warnings… (yellow, one line each)                                      │
│                                          [ Save as custom config ]      │
└──────────────────────────────────────────────────────────────────────┘
```

- **Cascade**: family change → uses the `/api/families` data (fetched once on panel open, cached),
  fills quant options (only `available` variants), pre-fills description/tags, resets
  ctx/advanced. Quant change → rebuilds the **context dropdown from that variant's `ctx_options`**
  (per-variant), pre-selects the first (native) option, shows/hides engine-specific advanced
  fields (KV cache, template list, thinking, reasoning visibility).
- **Context**: options from `ctx_options` + "Custom…"; selecting a YaRN option shows its `note`;
  the numeric input always carries the warning "not a verified context length for this model".
- **Docs**: `docs ↗` link on the family label row only when `docs_url` is set.
- **Save**: POST → on 400, field error shown inline (red) under the relevant field and no close;
  on success, toast "Saved — `<name>`" (3 s), panel keeps state, `refresh()` runs immediately so the
  row is visible without waiting for the 5 s poll. Success case resets Description to placeholder.
- Field visibility matrix (per engine): vLLM → hide KV cache; hide repeat-penalty **for the server**,
  but per the approved design the 3 sampling fields stay visible and the builder shows the note
  "applied via your opencode client, not the server" (value still stored in the config);
  llama_cpp.server → hide template/thinking/reasoning/KV/YaRN.

### Table changes (custom rows)

- `buildRow`: when `m.custom`, the name cell shows a small `custom` badge after the name and a small
  trash button (✕ styled like the star, grey → red on hover). `th` layout/widths unchanged (badges
  live inside existing cells).
- Trash → `confirm("Delete '<name>'? Removes the config and its opencode entry; stops it if running.")`
  → `DELETE /api/custom-model/{id}` → `refresh()`.
- Everything else (start/stop, offload, logs, favorites, search, sort, all-logs panel) works
  unchanged because rows are driven by `/api/status`.

## 9. Error handling

| case | behaviour |
|---|---|
| family/variant missing or weights gone | 400 with field message; quant dropdown greys out variants with `available:false` and a "(weights missing)" label |
| ctx out of range / non-integer | 400 |
| duplicate (family,variant,ctx) | 400 "already exists as `<id>`" |
| no free port 8100–8199 | 500 with message; nothing written |
| `custom_models.json` corrupt | startup log, built-ins only; next successful save rewrites the file (old file kept as `.corrupt-<ts>`) |
| opencode config missing / unreadable | save succeeds, `opencode_warning` in response, yellow note in toast area; file is never created fresh (only patched when it exists) |
| JSON write (the commit point) fails | model entry NOT saved; 500 with reason; no partial state. (Opencode-patch failure is the soft "warning" row above, never an error — the patch runs after the commit point.) |
| delete while running | stop first (existing stop path), then remove |
| two saves racing | `op_lock` serialises; second sees the duplicate and gets 400 |

Note: a built-in model entry can never be deleted through this API; the JSON only ever contains
custom entries, and the delete route checks `MODELS[id].get("custom")`.

## 10. Verification (no VRAM, no models started)

1. `python -m py_compile dashboard.py`; extract the `HTML_PAGE` `<script>` block to a temp file and
   `node --check` it.
2. **Throwaway instance** on 127.0.0.1:8099 (import `dashboard` in-process, run uvicorn on an
   alternate port; the 60 s power-sampler thread is harmless). Env overrides point both config
   files at scratch copies under `/tmp`: a copy of the real
   `~/.config/opencode/config.json` and a fresh scratch `custom_models.json`.
3. Against the scratch instance:
   - `GET /api/families`: qwen38-27b lists `Q8_K_XL` and `Q6_K_XL` variants with sizes; no
     non-existent quant appears; fable-amd lists exactly `IQ4_XS`; a family with weights moved
     away (temporarily rename one gguf in a scratch copy of a small family dir — or simulate with a
     metadata `base_dir` pointed at an empty scratch dir) shows `available: false`.
   - `POST` with bad family / missing variant / duplicate → 400 with the expected `field`.
   - Valid `POST` (qwen38 q8, 262144, custom temp 0.6, thinking false, sharp template):
     resulting `cmd` matches §6 byte-for-byte; port = 8100; scratch opencode file gained the
     provider + model with `limit.context 262144` and `temperature 0.6` only (no `topP` since
     default); a `.bak-*` file exists; `GET /api/status` shows the row with `custom: true`.
   - Second POST while the first still exists → port 8101; after `DELETE` of the first, a third
     POST → port reuses 8100 (first-free semantics).
   - `DELETE`: row gone from `/api/status`, scratch opencode model block gone (provider gone if it
     has no other models), JSON array empty.
   - Corrupt-JSON boot test: write garbage into the scratch JSON, start a second instance, confirm
     it serves `/api/status` with built-ins only and logs the error.
4. Kill the throwaway instance, remove scratch files. **Nothing on :80 is restarted, nothing is
   launched, GPU stays at idle draw.**
5. Verification of the two open assumptions (record outcome in the implementation plan's done
   criteria): (a) opencode forwards `repetitionPenalty` through
   `@ai-sdk/openai-compatible`; (b) each curated `docs_url` returns HTTP 200 (verify with `curl -sI`;
   drop any that don't).

## 11. Change summary (files touched)

| file | change |
|---|---|
| `dashboard.py` | `MODEL_FAMILIES` + disk-scan helpers, custom-model load/save/delete, 3 new endpoints, `custom` field in `/api/status`, offload-detection fix, builder panel HTML/CSS/JS, `buildRow` custom badge + delete |
| `custom_models.json` | new, created on first save |
| `~/.config/opencode/config.json` | patched at save/delete time (backed up each time) |

No new Python/JS dependencies; no systemd changes; built-in model entries untouched (except the
one-line offload-detection fix).

## 12. Open items deliberately left out of v1

- Editing a custom config in place (delete + re-create).
- Multi-GPU / tensor-parallel, speculative flags other than the qwen3.8 MTP draft, tool-call
  parsers in generated vLLM cmds.
- Auto-probing running servers to derive capabilities (approach C, rejected).
- Builder support for the chat-script / transformers-CLI engines.
