import subprocess
import signal
import socket
import os
from fastapi import FastAPI
from fastapi.responses import HTMLResponse, JSONResponse
import uvicorn

MODELS = {
    # --- Image Generation ---
    "janus": {
        "name": "Janus Pro 7B",
        "description": "Multimodal understanding & text-to-image generation",
        "port": 4343,
        "cmd": ["/usr/bin/python3", "app.py"],
        "cwd": "/mnt/raid1_sata/JanusPro7b",
        "env": {"CUDA_LAUNCH_BLOCKING": "1"},
        "protocol": "https",
        "category": "Image",
        "icon": "eye",
        "color": "#667eea",
        "tags": ["text-to-image", "image-to-text"],
        "supports_offload": True,
    },
    "flux": {
        "name": "FLUX Klein 9B",
        "description": "Fast image generation & style transfer",
        "port": 4242,
        "cmd": ["/mnt/raid1_sata/JanusPro7b/flux-venv/bin/python", "flux_app.py"],
        "cwd": "/mnt/raid1_sata/JanusPro7b",
        "env": {},
        "protocol": "https",
        "category": "Image",
        "icon": "image",
        "color": "#f5576c",
        "tags": ["text-to-image", "image-to-image"],
        "supports_offload": True,
    },
    "qwen_edit_general": {
        "name": "Qwen Image Edit (General)",
        "description": "General purpose image editing: upload + prompt → edited result (Qwen/Qwen-Image-Edit-2511)",
        "port": 4041,
        "cmd": ["/mnt/raid1_sata/JanusPro7b/qwen-venv/bin/python", "qwen_edit_general.py"],
        "cwd": "/mnt/raid1_sata/JanusPro7b",
        "env": {},
        "protocol": "https",
        "category": "Image",
        "icon": "wand-sparkles",
        "color": "#10b981",
        "tags": ["image-to-image", "image-editing"],
        "supports_offload": True,
    },
    "qwen_edit_dual": {
        "name": "Qwen Image Edit (Dual)",
        "description": "Multi-image fusion: combine 2 images with prompt guidance (person merging, scene fusion)",
        "port": 4042,
        "cmd": ["/mnt/raid1_sata/JanusPro7b/qwen-venv/bin/python", "qwen_edit_dual.py"],
        "cwd": "/mnt/raid1_sata/JanusPro7b",
        "env": {},
        "protocol": "https",
        "category": "Image",
        "icon": "users",
        "color": "#14b8a6",
        "tags": ["image-to-image", "image-fusion", "multi-image"],
        "supports_offload": True,
    },
    "qwen_edit_angles": {
        "name": "Qwen Image Edit (Angles)",
        "description": "Multi-angle camera control for 3D object rotation (specialized LoRA)",
        "port": 4040,
        "cmd": ["/mnt/raid1_sata/JanusPro7b/qwen-venv/bin/python", "qwen_app.py"],
        "cwd": "/mnt/raid1_sata/JanusPro7b",
        "env": {},
        "protocol": "https",
        "category": "Image",
        "icon": "rotate-3d",
        "color": "#43e97b",
        "tags": ["image-to-image", "3d"],
        "supports_offload": True,
    },
    "flux2_dev": {
        "name": "FLUX.2-dev",
        "description": "Black Forest Labs FLUX.2-dev - State-of-the-art text-to-image generation",
        "port": 9999,
        "cmd": ["/mnt/raid1_sata/nano/venv/bin/python", "flux_server.py"],
        "cwd": "/mnt/raid1_sata/nano",
        "env": {},
        "protocol": "http",
        "category": "Image",
        "icon": "sparkles",
        "color": "#fb923c",
        "tags": ["text-to-image", "sota"],
        "supports_offload": False,
    },
    # --- Video Generation ---
    "ltx": {
        "name": "LTX-2 Video",
        "description": "Audio-video generation from text and images (19B)",
        "port": 4141,
        "cmd": ["/mnt/raid1_sata/JanusPro7b/ltx-venv/bin/python", "ltx_app.py"],
        "cwd": "/mnt/raid1_sata/JanusPro7b",
        "env": {"PYTORCH_CUDA_ALLOC_CONF": "expandable_segments:True"},
        "protocol": "https",
        "category": "Video",
        "icon": "film",
        "color": "#a18cd1",
        "tags": ["text-to-video", "image-to-video"],
        "supports_offload": True,
    },
    "wan": {
        "name": "WAN 2.2 TI2V 5B",
        "description": "Text/image-to-video generation",
        "port": 3333,
        "cmd": ["/mnt/raid1_sata/wan2.2-TI2V-5B/venv/bin/python", "webui.py", "--port", "3333"],
        "cwd": "/mnt/raid1_sata/wan2.2-TI2V-5B",
        "env": {},
        "protocol": "http",
        "category": "Video",
        "icon": "clapperboard",
        "color": "#e879f9",
        "tags": ["text-to-video", "image-to-video"],
    },
    "video_gen": {
        "name": "Video Models (I2V)",
        "description": "Image-to-video with multiple models (SVD, I2VGen, etc.)",
        "port": 7860,
        "cmd": ["/mnt/raid1_sata/video/venv/bin/python", "simple_app.py"],
        "cwd": "/mnt/raid1_sata/video",
        "env": {},
        "protocol": "http",
        "category": "Video",
        "icon": "video",
        "color": "#f472b6",
        "tags": ["image-to-video"],
    },
    "echomimic_v2": {
        "name": "EchoMimicV2 Talking Head",
        "description": "Portrait image + audio → animated talking head video (768×768, body motion)",
        "port": 7867,
        "cmd": ["/mnt/raid1_sata/echomimic_v2/venv/bin/python", "/mnt/raid1_sata/echomimic_v2/app_talking_head.py"],
        "cwd": "/mnt/raid1_sata/echomimic_v2",
        "env": {
            "PYTHONPATH": "/mnt/raid1_sata/echomimic_v2/src",
            "PYTORCH_ALLOC_CONF": "expandable_segments:True",
        },
        "protocol": "http",
        "category": "Video",
        "icon": "video",
        "color": "#ec4899",
        "tags": ["talking-head", "image-to-video", "voice"],
        "supports_offload": False,
    },
    "ltx23": {
        "name": "LTX-2.3 Video (NEW)",
        "description": "Latest Lightricks audio-video gen (22B distilled, FP8)",
        "port": 7866,
        "cmd": ["/mnt/raid1_sata/video/venv/bin/python", "/mnt/raid1_sata/video/run_ltx23.py"],
        "cwd": "/mnt/raid1_sata/video",
        "env": {
            "PYTHONPATH": "/mnt/raid1_sata/video/LTX-2",
            "PYTORCH_ALLOC_CONF": "expandable_segments:True"
        },
        "protocol": "http",
        "category": "Video",
        "icon": "film",
        "color": "#d946ef",
        "tags": ["text-to-video", "image-to-video", "audio-video"],
        "supports_offload": False,
    },
    # --- Audio / Voice ---
    "rvc": {
        "name": "RVC Audio",
        "description": "Real-time voice conversion & TTS API",
        "port": 8080,
        "cmd": ["/mnt/raid1_sata/vcs-audio-5.0.0/venv/bin/python", "-m", "src.api.main"],
        "cwd": "/mnt/raid1_sata/vcs-audio-5.0.0",
        "env": {"SERVER_PORT": "8080"},
        "protocol": "http",
        "path": "/app",
        "category": "Audio",
        "icon": "mic",
        "color": "#22d3ee",
        "tags": ["audio-to-audio", "text-to-audio"],
    },
    "higgs": {
        "name": "Higgs Audio Voice LLM",
        "description": "Voice-based LLM with Whisper speech recognition",
        "port": 8888,
        "cmd": ["/mnt/raid1_sata/voice/bin/python3", "server.py"],
        "cwd": "/mnt/raid1_sata/voice/higgs-audio",
        "env": {},
        "protocol": "https",
        "category": "Audio",
        "icon": "audio-waveform",
        "color": "#2dd4bf",
        "tags": ["audio-to-text", "text-to-audio"],
    },
    "musicgen": {
        "name": "MusicGen (Meta)",
        "description": "Text-to-music generation with ChatTTS — facebook/musicgen-small",
        "port": 7870,
        "cmd": ["/mnt/raid1_sata/music/bin/python", "web_ui.py"],
        "cwd": "/mnt/raid1_sata/music",
        "env": {},
        "protocol": "http",
        "category": "Audio",
        "icon": "music",
        "color": "#f59e0b",
        "tags": ["text-to-audio", "music"],
    },
    "acestep": {
        "name": "ACE-Step 1.5",
        "description": "Full song generation from text/lyrics — hybrid LM+DiT, <2s on A100",
        "port": 7871,
        "cmd": ["/mnt/raid1_sata/acestep/venv/bin/acestep",
                "--port", "7871", "--server_name", "0.0.0.0",
                "--bf16", "true", "--cpu_offload", "true"],
        "cwd": "/mnt/raid1_sata/acestep",
        "env": {},
        "protocol": "http",
        "category": "Audio",
        "icon": "waveform",
        "color": "#ec4899",
        "tags": ["text-to-audio", "music", "lyrics"],
    },
    "stable_audio": {
        "name": "Stable Audio Open 1.0",
        "description": "Latent diffusion text-to-audio — stereo 44.1kHz, up to 47s (Stability AI)",
        "port": 7872,
        "cmd": ["/mnt/raid1_sata/stable-audio/venv/bin/python", "app.py"],
        "cwd": "/mnt/raid1_sata/stable-audio",
        "env": {},
        "protocol": "http",
        "category": "Audio",
        "icon": "radio",
        "color": "#8b5cf6",
        "tags": ["text-to-audio", "music", "sound-design"],
    },
    # --- LLM Inference ---
    "vllm_qwen72b": {
        "name": "Qwen 2.5 72B",
        "description": "vLLM OpenAI-compatible API (72B instruct)",
        "port": 8001,
        "cmd": ["/mnt/raid1_sata/vllm-env/bin/vllm", "serve", "Qwen/Qwen2.5-72B-Instruct",
                "--port", "8001", "--host", "0.0.0.0",
                "--enable-auto-tool-choice", "--tool-call-parser", "hermes",
                "--tensor-parallel-size", "1", "--gpu-memory-utilization", "0.95",
                "--max-model-len", "32768"],
        "cwd": "/mnt/raid1_sata/vllm-servers",
        "env": {},
        "protocol": "http",
        "category": "LLM",
        "icon": "brain",
        "color": "#818cf8",
        "tags": ["text-to-text"],
        "supports_offload": True,
    },
    "vllm_qwen_coder": {
        "name": "Qwen 2.5 Coder 32B",
        "description": "vLLM OpenAI-compatible API (code-specialized)",
        "port": 8002,
        "cmd": ["/mnt/raid1_sata/vllm-env/bin/vllm", "serve", "Qwen/Qwen2.5-Coder-32B-Instruct",
                "--port", "8002", "--host", "0.0.0.0",
                "--enable-auto-tool-choice", "--tool-call-parser", "hermes",
                "--tensor-parallel-size", "1", "--gpu-memory-utilization", "0.95",
                "--max-model-len", "32768"],
        "cwd": "/mnt/raid1_sata/vllm-servers",
        "env": {},
        "protocol": "http",
        "category": "LLM",
        "icon": "code",
        "color": "#a78bfa",
        "tags": ["text-to-text", "text-to-code"],
        "supports_offload": True,
    },
    "vllm_llama70b": {
        "name": "Llama 3.3 70B",
        "description": "vLLM OpenAI-compatible API (Meta Llama 70B)",
        "port": 8003,
        "cmd": ["/mnt/raid1_sata/vllm-env/bin/vllm", "serve", "meta-llama/Llama-3.3-70B-Instruct",
                "--port", "8003", "--host", "0.0.0.0",
                "--enable-auto-tool-choice", "--tool-call-parser", "llama3_json",
                "--tensor-parallel-size", "1", "--gpu-memory-utilization", "0.95",
                "--max-model-len", "32768"],
        "cwd": "/mnt/raid1_sata/vllm-servers",
        "env": {},
        "protocol": "http",
        "category": "LLM",
        "icon": "cpu",
        "color": "#c084fc",
        "tags": ["text-to-text"],
        "supports_offload": True,
    },
    "vllm_deepseek32b": {
        "name": "DeepSeek R1 32B",
        "description": "vLLM OpenAI-compatible API (DeepSeek reasoning)",
        "port": 8004,
        "cmd": ["/mnt/raid1_sata/vllm-env/bin/vllm", "serve", "deepseek-ai/DeepSeek-R1-Distill-Qwen-32B",
                "--port", "8004", "--host", "0.0.0.0",
                "--enable-auto-tool-choice", "--tool-call-parser", "hermes",
                "--tensor-parallel-size", "1", "--gpu-memory-utilization", "0.95",
                "--max-model-len", "65536"],
        "cwd": "/mnt/raid1_sata/vllm-servers",
        "env": {},
        "protocol": "http",
        "category": "LLM",
        "icon": "zap",
        "color": "#e879f9",
        "tags": ["text-to-text"],
        "supports_offload": True,
    },
    "vllm_deepseek70b": {
        "name": "DeepSeek R1 70B",
        "description": "vLLM OpenAI-compatible API (DeepSeek large reasoning)",
        "port": 8005,
        "cmd": ["/mnt/raid1_sata/vllm-env/bin/vllm", "serve", "deepseek-ai/DeepSeek-R1-Distill-Llama-70B",
                "--port", "8005", "--host", "0.0.0.0",
                "--enable-auto-tool-choice", "--tool-call-parser", "hermes",
                "--tensor-parallel-size", "1", "--gpu-memory-utilization", "0.95",
                "--max-model-len", "32768"],
        "cwd": "/mnt/raid1_sata/vllm-servers",
        "env": {},
        "protocol": "http",
        "category": "LLM",
        "icon": "flame",
        "color": "#f472b6",
        "tags": ["text-to-text"],
        "supports_offload": True,
    },
    "vllm_qwen35b": {
        "name": "Qwen 3.5 35B-A3B",
        "description": "Chat UI + vLLM API (MoE, 3B active of 35B)",
        "port": 8007,
        "cmd": ["/mnt/raid1_sata/vllm-env/bin/python", "qwen35b_chat.py"],
        "cwd": "/mnt/raid1_sata/JanusPro7b",
        "env": {},
        "protocol": "https",
        "category": "LLM",
        "icon": "sparkles",
        "color": "#38bdf8",
        "tags": ["text-to-text", "text-to-code"],
        "supports_offload": True,
    },
    "vllm_qwen30b_abliterated": {
        "name": "Qwen 3 30B Abliterated",
        "description": "Chat UI with Transformers (uncensored MoE, 3B active of 30B)",
        "port": 8008,
        "cmd": ["/mnt/raid1_sata/vllm-env/bin/python", "qwen30b_abliterated_transformers.py"],
        "cwd": "/mnt/raid1_sata/JanusPro7b",
        "env": {},
        "protocol": "https",
        "category": "LLM",
        "icon": "unlock",
        "color": "#fb7185",
        "tags": ["text-to-text", "uncensored"],
        "supports_offload": False,
    },
    "vllm_qwen35_opus_reasoning": {
        "name": "Qwen 3.5 27B Claude Opus Reasoning",
        "description": "vLLM OpenAI-compatible API (Claude 4.6 Opus reasoning distilled)",
        "port": 8009,
        "cmd": ["/mnt/raid1_sata/vllm-env/bin/vllm", "serve", "Jackrong/Qwen3.5-27B-Claude-4.6-Opus-Reasoning-Distilled",
                "--port", "8009", "--host", "0.0.0.0",
                "--enable-auto-tool-choice", "--tool-call-parser", "hermes",
                "--tensor-parallel-size", "1", "--gpu-memory-utilization", "0.95",
                "--max-model-len", "32768"],
        "cwd": "/mnt/raid1_sata/vllm-servers",
        "env": {},
        "protocol": "http",
        "category": "LLM",
        "icon": "brain-circuit",
        "color": "#a855f7",
        "tags": ["text-to-text", "reasoning"],
        "supports_offload": True,
    },

    "qwen36_35b_chat": {
        "name": "Qwen 3.6 35B-A3B Chat",
        "description": "Gradio chat UI for Qwen 3.6 35B-A3B — thinking mode toggle, streaming, tok/s stats",
        "port": 8014,
        "cmd": ["/mnt/raid1_sata/vllm-env/bin/python", "qwen36_35b_chat.py"],
        "cwd": "/mnt/raid1_sata/JanusPro7b",
        "env": {},
        "protocol": "https",
        "category": "LLM",
        "icon": "message-square-code",
        "color": "#48cae4",
        "tags": ["text-to-text", "text-to-code"],
        "supports_offload": False,
    },
    "vllm_qwen36_27b": {
        "name": "Qwen 3.6 27B",
        "description": "vLLM OpenAI-compatible API (Qwen3.6 27B BF16, full precision — coding & chat)",
        "port": 8006,
        "cmd": ["/mnt/raid1_sata/vllm-env/bin/vllm", "serve", "/mnt/raid1_sata/models/qwen36-27b",
                "--port", "8006", "--host", "0.0.0.0",
                "--enable-auto-tool-choice", "--tool-call-parser", "hermes",
                "--tensor-parallel-size", "1", "--gpu-memory-utilization", "0.90",
                "--max-model-len", "131072", "--trust-remote-code"],
        "env": {"PYTORCH_ALLOC_CONF": "expandable_segments:True"},
        "cwd": "/mnt/raid1_sata/vllm-servers",
        "protocol": "http",
        "category": "LLM",
        "icon": "zap",
        "color": "#f59e0b",
        "tags": ["text-to-text", "text-to-code"],
        "supports_offload": True,
    },
    "vllm_qwen36_35b": {
        "name": "Qwen 3.6 35B-A3B",
        "description": "vLLM OpenAI-compatible API (Qwen3.6 MoE hybrid-attn, 3B active of 35B — coding & chat)",
        "port": 8012,
        "cmd": ["/mnt/raid1_sata/vllm-env/bin/vllm", "serve", "/mnt/raid1_sata/models/qwen36-35b",
                "--port", "8012", "--host", "0.0.0.0",
                "--enable-auto-tool-choice", "--tool-call-parser", "hermes",
                "--tensor-parallel-size", "1", "--gpu-memory-utilization", "0.92",
                "--max-model-len", "131072", "--trust-remote-code",
                "--enforce-eager"],
        "env": {"PYTORCH_ALLOC_CONF": "expandable_segments:True"},
        "cwd": "/mnt/raid1_sata/vllm-servers",
        "protocol": "http",
        "category": "LLM",
        "icon": "terminal",
        "color": "#48cae4",
        "tags": ["text-to-text", "text-to-code"],
        "supports_offload": True,
    },
    "qwen3_coder_next_q4": {
        "name": "Qwen3-Coder-Next Q4",
        "description": "llama.cpp — 80B MoE (3B active), Q4_K_M, 131K ctx — agent/tool use",
        "port": 8085,
        "systemd_service": "qwen3-coder-next-q4.service",
        "cmd": ["/mnt/raid1_sata/models/ik_llama.cpp/build/bin/llama-server",
                "--model", "/mnt/raid1_sata/models/qwen3-coder-next/Qwen3-Coder-Next-Q4_K_M/Qwen3-Coder-Next-Q4_K_M-00001-of-00004.gguf",
                "--alias", "qwen3-coder-next",
                "--ctx-size", "131072",
                "-ngl", "99",
                "-b", "2048", "-ub", "2048",
                "--host", "0.0.0.0",
                "--port", "8085",
                "--jinja",
                "--threads", "16",
                "--reasoning-tokens", "none"],
        "cwd": "/mnt/raid1_sata/models/ik_llama.cpp",
        "env": {},
        "protocol": "http",
        "category": "LLM",
        "icon": "code-2",
        "color": "#10b981",
        "tags": ["text-to-text", "text-to-code", "agents"],
        "supports_offload": False,
    },
    "qwen3_coder_next": {
        "name": "Qwen3-Coder-Next",
        "description": "llama.cpp — 80B MoE (3B active), 131K ctx, SWE-Bench 70.6 — best agent/tool use",
        "port": 8084,
        "systemd_service": "qwen3-coder-next.service",
        "cmd": ["/mnt/raid1_sata/models/ik_llama.cpp/build/bin/llama-server",
                "--model", "/mnt/raid1_sata/models/qwen3-coder-next/Qwen3-Coder-Next-Q6_K/Qwen3-Coder-Next-Q6_K-00001-of-00004.gguf",
                "--alias", "qwen3-coder-next",
                "--ctx-size", "131072",
                "-ngl", "99",
                "-b", "2048", "-ub", "2048",
                "--host", "0.0.0.0",
                "--port", "8084",
                "--jinja",
                "--threads", "16",
                "--reasoning-tokens", "none"],
        "cwd": "/mnt/raid1_sata/models/ik_llama.cpp",
        "env": {},
        "protocol": "http",
        "category": "LLM",
        "icon": "code-2",
        "color": "#10b981",
        "tags": ["text-to-text", "text-to-code", "agents"],
        "supports_offload": False,
    },
    "qwen36_35b_uncensored": {
        "name": "Qwen3.6 35B Uncensored (Q8)",
        "description": "llama.cpp — Q8_K_P, 43GB, Qwen3.6 35B-A3B uncensored, 256K context",
        "port": 8086,
        "systemd_service": "qwen36-35b-uncensored.service",
        "cmd": ["/mnt/raid1_sata/models/ik_llama.cpp/start-qwen36-35b-uncensored.sh"],
        "cwd": "/mnt/raid1_sata/models/ik_llama.cpp",
        "env": {},
        "protocol": "http",
        "category": "LLM",
        "icon": "zap",
        "color": "#f97316",
        "tags": ["text-to-text", "text-to-code", "agents"],
        "supports_offload": False,
    },
    "qwen36_35b_uncensored_q6": {
        "name": "Qwen3.6 35B Uncensored (Q6)",
        "description": "llama.cpp — Q6_K_P, 31GB, Qwen3.6 35B-A3B uncensored, 256K context, pairs with Coder-Next",
        "port": 8087,
        "cmd": ["/mnt/raid1_sata/models/ik_llama.cpp/start-qwen36-35b-uncensored-q6.sh"],
        "cwd": "/mnt/raid1_sata/models/ik_llama.cpp",
        "env": {},
        "protocol": "http",
        "category": "LLM",
        "icon": "zap",
        "color": "#fb923c",
        "tags": ["text-to-text", "text-to-code", "agents"],
        "supports_offload": False,
    },
    "qwen35_122b_q5": {
        "name": "Qwen3.5 122B Q5 (single)",
        "description": "llama.cpp — Q5_K_M, 88GB, max quality, single session",
        "port": 8082,
        "systemd_service": "qwen35-122b.service",
        "cmd": ["/mnt/raid1_sata/models/ik_llama.cpp/build/bin/llama-server",
                "--model", "/mnt/raid1_sata/models/qwen35-122b/Qwen_Qwen3.5-122B-A10B-Q5_K_M/Qwen_Qwen3.5-122B-A10B-Q5_K_M-00001-of-00003.gguf",
                "--alias", "qwen35-122b",
                "--ctx-size", "262144",
                "-ngl", "99",
                "-b", "2048", "-ub", "2048",
                "--host", "0.0.0.0",
                "--port", "8082",
                "--jinja",
                "--threads", "16"],
        "cwd": "/mnt/raid1_sata/models/ik_llama.cpp",
        "env": {},
        "protocol": "http",
        "category": "LLM",
        "icon": "brain",
        "color": "#22d3ee",
        "tags": ["text-to-text", "text-to-code", "agents"],
        "supports_offload": False,
    },
    "qwen35_122b_q4": {
        "name": "Qwen3.5 122B Q4 (multi-agent)",
        "description": "llama.cpp — Q4_K_M, 75GB, 4 parallel agents, 256K ctx each",
        "port": 8083,
        "systemd_service": "qwen35-122b-q4.service",
        "cmd": ["/mnt/raid1_sata/models/ik_llama.cpp/build/bin/llama-server",
                "--model", "/mnt/raid1_sata/models/qwen35-122b/Qwen_Qwen3.5-122B-A10B-Q4_K_M/Qwen_Qwen3.5-122B-A10B-Q4_K_M-00001-of-00002.gguf",
                "--alias", "qwen35-122b",
                "--ctx-size", "262144",
                "--parallel", "4",
                "-ngl", "99",
                "-b", "2048", "-ub", "2048",
                "--host", "0.0.0.0",
                "--port", "8083",
                "--jinja",
                "--threads", "16"],
        "cwd": "/mnt/raid1_sata/models/ik_llama.cpp",
        "env": {},
        "protocol": "http",
        "category": "LLM",
        "icon": "cpu",
        "color": "#a78bfa",
        "tags": ["text-to-text", "text-to-code", "agents"],
        "supports_offload": False,
    },
    "minimax_m25": {
        "name": "MiniMax M2.5",
        "description": "llama.cpp OpenAI-compatible API (230B MoE, 10B active, coding/agents)",
        "port": 8081,
        "cmd": ["/mnt/raid1_sata/models/ik_llama.cpp/build/bin/llama-server",
                "--model", "/mnt/raid1_sata/models/minimax-m2.5/smol-IQ3_KS/MiniMax-M2.5-smol-IQ3_KS-00001-of-00003.gguf",
                "--alias", "minimax-m2.5",
                "--ctx-size", "65536",
                "-ngl", "99",
                "-b", "2048", "-ub", "2048",
                "--host", "0.0.0.0",
                "--port", "8081",
                "--jinja",
                "--threads", "16"],
        "cwd": "/mnt/raid1_sata/models/ik_llama.cpp",
        "env": {},
        "protocol": "http",
        "category": "LLM",
        "icon": "rocket",
        "color": "#06b6d4",
        "tags": ["text-to-text", "text-to-code", "agents"],
        "supports_offload": False,
    },
    # --- Tools ---
    "llama_ppt": {
        "name": "Llama 3 PPT Generator",
        "description": "AI-powered presentation generator (requires Ollama)",
        "port": 5000,
        "cmd": ["/mnt/raid1_sata/llama/llama_env/bin/python", "app.py"],
        "cwd": "/mnt/raid1_sata/llama",
        "env": {},
        "protocol": "http",
        "category": "Tools",
        "icon": "presentation",
        "color": "#34d399",
        "tags": ["text-to-document"],
    },
    "open_webui": {
        "name": "Open WebUI (Ollama)",
        "description": "Chat interface for Ollama models",
        "port": 3000,
        "cmd": ["/snap/bin/open-webui", "serve", "--port", "3000"],
        "cwd": "/mnt/raid1_sata/olama",
        "env": {"PORT": "3000"},
        "protocol": "http",
        "category": "Tools",
        "icon": "message-circle",
        "color": "#60a5fa",
        "tags": ["text-to-text"],
    },
}

# Group ordering for display
CATEGORY_ORDER = ["Image", "Video", "Audio", "LLM", "Tools"]

processes: dict[str, subprocess.Popen] = {}
log_files: dict[str, object] = {}

app = FastAPI()

# ── Power tracking ────────────────────────────────────────────────
import threading, json as _json_mod, datetime as _dt_mod, csv

POWER_LOG = "/mnt/raid1_sata/JanusPro7b/logs/power_usage.json"
POWER_CSV = "/mnt/raid1_sata/JanusPro7b/logs/power_history.csv"

# Model-specific power profiles (watts) — estimated from nvidia-smi observations
# These represent the ADDITIONAL power draw above idle when a model is active
MODEL_POWER_PROFILES = {
    # llama.cpp models (GPU memory footprint drives power)
    "qwen35-122b": {"watts": 450, "idle_watts": 40, "desc": "122B Q5, ~88GB VRAM, single GPU"},
    "qwen35-122b-q4": {"watts": 500, "idle_watts": 50, "desc": "122B Q4, ~75GB VRAM, 4 parallel agents"},
    "qwen36-35b-uncensored": {"watts": 280, "idle_watts": 30, "desc": "35B Q8, ~43GB VRAM"},
    "qwen3-coder-next": {"watts": 250, "idle_watts": 25, "desc": "80B MoE Q6, ~70GB VRAM"},
    "qwen3-coder-next-q4": {"watts": 300, "idle_watts": 35, "desc": "80B MoE Q4, ~55GB VRAM, 4 parallel"},
    # vLLM models
    "qwen36-35b": {"watts": 400, "idle_watts": 35, "desc": "35B MoE vLLM"},
    "qwen36-27b": {"watts": 380, "idle_watts": 35, "desc": "27B vLLM"},
    "deepseek-ai/DeepSeek-R1-Distill-Qwen-32B": {"watts": 380, "idle_watts": 35, "desc": "32B vLLM"},
    # Image/Video models (gradio)
    "janus": {"watts": 350, "idle_watts": 30, "desc": "Janus Pro 7B image gen"},
    "flux": {"watts": 350, "idle_watts": 30, "desc": "FLUX Klein 9B"},
    "ltx": {"watts": 350, "idle_watts": 30, "desc": "LTX-2 Video"},
    # Idle GPU baseline
    "idle": {"watts": 25, "idle_watts": 25, "desc": "GPU idle baseline"},
}

# Map modelID from opencode DB to power profile key
def _model_to_power_key(model_id: str) -> str:
    """Map an opencode modelID to a power profile key."""
    if model_id in MODEL_POWER_PROFILES:
        return model_id
    # Try matching by substring
    for key in MODEL_POWER_PROFILES:
        if key in model_id or model_id in key:
            return key
    # Check for llama.cpp models by VRAM usage
    return "idle"  # fallback

def _load_power_log() -> dict:
    try:
        return _json_mod.loads(open(POWER_LOG).read())
    except Exception:
        return {}

def _save_power_log(data: dict):
    try:
        open(POWER_LOG, "w").write(_json_mod.dumps(data))
    except Exception:
        pass

def _append_power_csv(date_str: str, watts: float, model: str = "unknown"):
    """Append a power sample to CSV for historical backfill."""
    try:
        file_exists = os.path.isfile(POWER_CSV)
        with open(POWER_CSV, "a", newline="") as f:
            writer = csv.writer(f)
            if not file_exists:
                writer.writerow(["timestamp", "date", "power_watts", "model", "sample_type"])
            writer.writerow([
                _dt_mod.datetime.now().isoformat(),
                date_str,
                f"{watts:.1f}",
                model,
                "sample"
            ])
    except Exception:
        pass

def _power_sampler():
    """Sample GPU power every 30s, accumulate per-day, and archive completed days."""
    import time
    last_day = None
    
    while True:
        try:
            today = _dt_mod.date.today().isoformat()
            
            # Archive yesterday's data if we've moved to a new day
            if last_day and last_day != today:
                data = _load_power_log()
                yesterday_wh = data.pop(last_day, 0)
                if yesterday_wh > 0:
                    # Write yesterday to CSV for persistence
                    _append_power_csv(last_day, yesterday_wh * 60, "historical")
                _save_power_log(data)
            
            # Sample power
            out = subprocess.run(
                ["nvidia-smi", "--query-gpu=power.draw", "--format=csv,noheader,nounits"],
                capture_output=True, text=True, timeout=5,
            )
            watts = float(out.stdout.strip())
            
            data = _load_power_log()
            # watt-hours = watts * (1min / 60min)
            data[today] = data.get(today, 0) + watts / 60.0
            _save_power_log(data)
            
            # Also log to CSV for backfill
            _append_power_csv(today, watts, "live")
            
            last_day = today
        except Exception:
            pass
        time.sleep(30)

_power_thread = threading.Thread(target=_power_sampler, daemon=True)
_power_thread.start()


def _backfill_power_from_tokens() -> dict:
    """Backfill historical power data using token counts from opencode DB and model power profiles.
    
    Uses a two-tier approach:
    1. If overlap days exist (power data + token data), calibrate wh-per-mmt from real measurements
    2. Otherwise, estimate from model wattage × active hours (derived from token count / throughput)
       with a per-day cap to reflect realistic usage patterns
    """
    import sqlite3
    db_path = "/root/.local/share/opencode/opencode.db"
    try:
        conn = sqlite3.connect(f"file:{db_path}?mode=ro", uri=True)
        rows = conn.execute(
            "SELECT time_created, data FROM message WHERE json_extract(data, '$.role') = 'assistant'"
        ).fetchall()
        conn.close()
    except Exception:
        return {}
    
    # Build per-day, per-model token counts
    day_model_tokens: dict[str, dict[str, int]] = {}
    for time_created, data_str in rows:
        try:
            d = _json_mod.loads(data_str)
        except Exception:
            continue
        tokens = d.get("tokens") or {}
        tot = tokens.get("total", 0) or (tokens.get("input", 0) + tokens.get("output", 0))
        if tot == 0:
            continue
        model_id = d.get("modelID", "unknown")
        day = _dt_mod.datetime.fromtimestamp(time_created / 1000).strftime("%Y-%m-%d")
        
        day_model_tokens.setdefault(day, {})
        day_model_tokens[day][model_id] = day_model_tokens[day].get(model_id, 0) + tot
    
    power_data = _load_power_log()
    token_days = sorted(day_model_tokens.keys())
    
    # Find overlap days and compute calibrated wh-per-mmt ratios
    overlap_days = [d for d in token_days if d in power_data]
    recent_overlap = overlap_days[-14:] if len(overlap_days) >= 14 else overlap_days
    
    model_wh_per_mmt: dict[str, list[tuple[float, float]]] = {}
    
    for day in recent_overlap:
        day_tokens = day_model_tokens.get(day, {})
        if not day_tokens:
            continue
        day_wh = power_data.get(day, 0)
        if day_wh == 0:
            continue
        
        total_day_tokens = sum(day_tokens.values())
        
        for model_id, model_tokens in day_tokens.items():
            power_key = _model_to_power_key(model_id)
            profile = MODEL_POWER_PROFILES.get(power_key)
            if not profile:
                continue
            
            model_weighted = model_tokens * profile["watts"]
            total_weighted = sum(
                mt * MODEL_POWER_PROFILES.get(_model_to_power_key(mid), MODEL_POWER_PROFILES["idle"])["watts"]
                for mid, mt in day_tokens.items()
                if _model_to_power_key(mid) in MODEL_POWER_PROFILES
            )
            
            if total_weighted == 0:
                continue
            
            model_wh = (model_weighted / total_weighted) * day_wh
            model_mmt = model_tokens / 1e6
            
            model_wh_per_mmt.setdefault(power_key, [])
            model_wh_per_mmt[power_key].append((model_mmt, model_wh))
    
    # Average wh per million tokens per model type
    avg_wh_per_mmt: dict[str, float] = {}
    for key, samples in model_wh_per_mmt.items():
        total_wh = sum(s[1] for s in samples)
        total_mmt = sum(s[0] for s in samples)
        if total_mmt > 0 and len(samples) >= 3:
            # Need at least 3 overlap days for reliable calibration
            avg_wh_per_mmt[key] = total_wh / total_mmt
    
    import math as _math_mod
    
    # Token-per-minute estimates per model type
    def _tok_per_min(model_id: str) -> float:
        if "122b" in model_id.lower() or "122B" in model_id:
            return 800
        elif "35b" in model_id.lower() or "35B" in model_id:
            return 1500
        elif "27b" in model_id.lower() or "27B" in model_id:
            return 1200
        elif "32b" in model_id.lower() or "32B" in model_id:
            return 1000
        return 500
    
    # Estimate active hours per day based on token volume and model type
    def _estimate_daily_hours(model_id: str, tokens: int) -> float:
        """Estimate hours of GPU activity based on token count."""
        if tokens < 100000:
            return max(0.5, tokens / 100000 * 2)
        elif tokens < 1000000:
            return 2 + (tokens - 100000) / 900000 * 4
        elif tokens < 10000000:
            return 6 + (tokens - 1000000) / 9000000 * 6
        else:
            return min(18, 12 + (_math_mod.log2(max(1, tokens / 10000000)) * 3))
    
    # Backfill ALL days without real power data
    backfill = {}
    for day, day_tokens in day_model_tokens.items():
        if day in power_data:
            continue
        
        total_day_wh = 0
        max_active_hours = 0
        
        for model_id, model_tokens in day_tokens.items():
            power_key = _model_to_power_key(model_id)
            profile = MODEL_POWER_PROFILES.get(power_key)
            if not profile:
                continue
            
            # Use calibrated wh-per-mmt if we have overlap data
            if power_key in avg_wh_per_mmt and model_tokens > 0:
                model_wh = avg_wh_per_mmt[power_key] * (model_tokens / 1e6)
                total_day_wh += model_wh
                # Also estimate hours for idle baseline calculation
                hours = _estimate_daily_hours(model_id, model_tokens)
                max_active_hours = max(max_active_hours, hours)
            else:
                # Estimate from watts × hours + idle baseline
                hours = _estimate_daily_hours(model_id, model_tokens)
                max_active_hours = max(max_active_hours, hours)
                active_wh = hours * profile["watts"]
                total_day_wh += active_wh
        
        # Add idle baseline for remaining hours
        idle_hours = max(0, 24 - max_active_hours)
        total_day_wh += idle_hours * MODEL_POWER_PROFILES["idle"]["watts"]
        
        backfill[day] = total_day_wh
    
    return backfill


def get_all_power_data() -> dict:
    """Get combined real + backfilled power data."""
    real_data = _load_power_log()
    backfill_data = _backfill_power_from_tokens()
    
    # Merge: real data takes precedence
    all_data = {**backfill_data, **real_data}
    return all_data


def is_port_open(port: int) -> bool:
    try:
        with socket.create_connection(("127.0.0.1", port), timeout=1):
            return True
    except (ConnectionRefusedError, TimeoutError, OSError):
        return False


def get_model_status(model_id: str) -> dict:
    model = MODELS[model_id]
    proc = processes.get(model_id)
    managed = proc and proc.poll() is None
    port_up = is_port_open(model["port"])
    if managed:
        return {"status": "ready" if port_up else "starting", "pid": proc.pid, "managed": True}
    else:
        processes.pop(model_id, None)
        if port_up:
            return {"status": "ready", "pid": None, "managed": False}
        return {"status": "stopped", "pid": None, "managed": False}


@app.get("/api/power")
def api_power():
    """Return historical watt-hour usage per day (real + backfilled)."""
    data = get_all_power_data()
    sorted_data = dict(sorted(data.items()))
    today = _dt_mod.date.today().isoformat()
    today_wh = data.get(today, 0)
    # Add backfill info
    real_days = set(_load_power_log().keys())
    backfill_days = set(get_all_power_data().keys()) - real_days
    has_backfill = len(backfill_days) > 0
    return JSONResponse({
        "by_day": sorted_data,
        "today_wh": today_wh,
        "total_wh": sum(sorted_data.values()),
        "total_cost_estimate": 0,  # computed client-side
        "has_backfill": has_backfill,
        "backfill_days": len(backfill_days),
        "real_days": len(real_days),
    })


@app.get("/api/tokens")
def api_tokens(days: int = 30):
    """Query opencode SQLite DB for token usage stats."""
    import sqlite3, json as _json, datetime
    db_path = "/root/.local/share/opencode/opencode.db"
    try:
        conn = sqlite3.connect(f"file:{db_path}?mode=ro", uri=True)
        rows = conn.execute(
            "SELECT time_created, data FROM message WHERE json_extract(data, '$.role') = 'assistant'"
        ).fetchall()
        conn.close()
    except Exception as e:
        return JSONResponse({"error": str(e)}, status_code=500)

    by_day: dict = {}
    by_model: dict = {}
    total_input = total_output = total_tokens = 0

    for time_created, data_str in rows:
        try:
            d = _json.loads(data_str)
        except Exception:
            continue
        tokens = d.get("tokens") or {}
        inp = tokens.get("input", 0) or 0
        out = tokens.get("output", 0) or 0
        tot = tokens.get("total", inp + out) or (inp + out)
        model_id = d.get("modelID", "unknown")
        day = datetime.datetime.fromtimestamp(time_created / 1000).strftime("%Y-%m-%d")

        total_input += inp
        total_output += out
        total_tokens += tot

        by_model.setdefault(model_id, {"input": 0, "output": 0, "total": 0})
        by_model[model_id]["input"] += inp
        by_model[model_id]["output"] += out
        by_model[model_id]["total"] += tot

        by_day.setdefault(day, {"input": 0, "output": 0, "total": 0})
        by_day[day]["input"] += inp
        by_day[day]["output"] += out
        by_day[day]["total"] += tot

    limit = None if days <= 0 else days
    sorted_days = dict(sorted(by_day.items())[-limit:] if limit else sorted(by_day.items()))
    sorted_models = dict(sorted(by_model.items(), key=lambda x: -x[1]["total"])[:10])

    return JSONResponse({
        "total_input": total_input,
        "total_output": total_output,
        "total_tokens": total_tokens,
        "by_day": sorted_days,
        "by_model": sorted_models,
    })


@app.get("/api/gpu")
def api_gpu():
    try:
        out = subprocess.run(
            ["nvidia-smi", "--query-gpu=memory.used,memory.total,utilization.gpu,temperature.gpu,power.draw,power.limit,name",
             "--format=csv,noheader,nounits"],
            capture_output=True, text=True, timeout=5,
        )
        parts = [p.strip() for p in out.stdout.strip().split(",")]
        return JSONResponse({
            "vram_used": int(parts[0]),
            "vram_total": int(parts[1]),
            "gpu_util": int(parts[2]),
            "temp": int(parts[3]),
            "power": float(parts[4]),
            "power_limit": float(parts[5]),
            "name": parts[6],
        })
    except Exception:
        return JSONResponse({"error": "nvidia-smi failed"}, status_code=500)


@app.get("/api/status")
def api_status():
    result = {}
    for model_id, model in MODELS.items():
        result[model_id] = {
            "name": model["name"],
            "description": model["description"],
            "port": model["port"],
            "protocol": model["protocol"],
            "category": model["category"],
            "icon": model["icon"],
            "color": model["color"],
            "tags": model.get("tags", []),
            "path": model.get("path", ""),
            "supports_offload": model.get("supports_offload", False),
            **get_model_status(model_id),
        }
    return JSONResponse(result)


def stop_other_llm_models(exclude_model_id: str):
    """Stop all running LLM and Video models except the one being started."""
    for mid, m in MODELS.items():
        if mid == exclude_model_id or m["category"] not in ("LLM", "Video"):
            continue
        proc = processes.get(mid)
        if proc and proc.poll() is None:
            try:
                os.killpg(os.getpgid(proc.pid), signal.SIGKILL)
                proc.wait()
            except Exception:
                pass
            processes.pop(mid, None)
            if mid in log_files:
                try:
                    log_files[mid].close()
                except Exception:
                    pass
                log_files.pop(mid, None)
        # Also kill anything on the port
        port = m["port"]
        if is_port_open(port):
            kill_port(port)


@app.post("/api/start/{model_id}")
def api_start(model_id: str, offload: bool = False):
    if model_id not in MODELS:
        return JSONResponse({"error": "Unknown model"}, status_code=404)

    status = get_model_status(model_id)
    if status["status"] != "stopped":
        return JSONResponse({"message": "Already running", **status})

    model = MODELS[model_id]

    # Auto-stop other LLM/Video models to free GPU memory
    if model["category"] in ("LLM", "Video"):
        stop_other_llm_models(model_id)

    # If managed by systemd, delegate start to systemctl
    systemd_service = model.get("systemd_service")
    if systemd_service:
        subprocess.run(["systemctl", "start", systemd_service], capture_output=True)
        return JSONResponse({"message": "Starting via systemd"})

    env = {**os.environ, **model["env"]}
    cmd = list(model["cmd"])

    if offload and model.get("supports_offload"):
        # vLLM models: add --cpu-offload-gb flag
        if cmd[0].endswith("vllm"):
            cmd.extend(["--cpu-offload-gb", "24"])
        else:
            # Gradio/other apps: set env var for the app to check
            env["CPU_OFFLOAD"] = "1"

    log_path = f"/mnt/raid1_sata/JanusPro7b/logs/{model_id}.log"
    os.makedirs(os.path.dirname(log_path), exist_ok=True)

    # Close previous log file handle if any
    if model_id in log_files:
        try:
            log_files[model_id].close()
        except Exception:
            pass

    lf = open(log_path, "w")
    log_files[model_id] = lf

    proc = subprocess.Popen(
        cmd,
        cwd=model["cwd"],
        env=env,
        stdout=lf,
        stderr=subprocess.STDOUT,
        preexec_fn=os.setsid,
    )
    processes[model_id] = proc
    return JSONResponse({"message": "Starting", "pid": proc.pid})


def kill_port(port: int):
    """Kill any process listening on the given port."""
    try:
        result = subprocess.run(
            ["fuser", f"{port}/tcp"], capture_output=True, text=True
        )
        for pid_str in result.stdout.split():
            pid = int(pid_str.strip())
            try:
                os.killpg(os.getpgid(pid), signal.SIGKILL)
            except (ProcessLookupError, PermissionError):
                try:
                    os.kill(pid, signal.SIGKILL)
                except (ProcessLookupError, PermissionError):
                    pass
    except Exception:
        pass


@app.post("/api/stop/{model_id}")
def api_stop(model_id: str):
    if model_id not in MODELS:
        return JSONResponse({"error": "Unknown model"}, status_code=404)

    model = MODELS[model_id]

    # If managed by systemd, stop the service so it doesn't restart
    systemd_service = model.get("systemd_service")
    if systemd_service:
        subprocess.run(["systemctl", "stop", systemd_service], capture_output=True)

    proc = processes.get(model_id)
    if proc and proc.poll() is None:
        try:
            os.killpg(os.getpgid(proc.pid), signal.SIGKILL)
            proc.wait()
        except Exception:
            pass
        processes.pop(model_id, None)
        if model_id in log_files:
            try:
                log_files[model_id].close()
            except Exception:
                pass
            log_files.pop(model_id, None)
        return JSONResponse({"message": "Stopped"})

    # Not managed by us — kill whatever is on the port
    processes.pop(model_id, None)
    port = model["port"]
    if is_port_open(port):
        kill_port(port)
        return JSONResponse({"message": "Stopped"})
    return JSONResponse({"message": "Already stopped"})


@app.get("/api/logs/{model_id}")
def api_logs(model_id: str):
    if model_id not in MODELS:
        return JSONResponse({"error": "Unknown model"}, status_code=404)
    log_path = f"/mnt/raid1_sata/JanusPro7b/logs/{model_id}.log"
    try:
        with open(log_path) as f:
            lines = f.readlines()
            return JSONResponse({"logs": "".join(lines[-200:])})
    except FileNotFoundError:
        return JSONResponse({"logs": ""})


@app.get("/", response_class=HTMLResponse)
def index():
    return HTML_PAGE


HTML_PAGE = r"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width, initial-scale=1.0">
<title>GPU Model Dashboard</title>
<style>
  @import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700;800&display=swap');
  * { margin: 0; padding: 0; box-sizing: border-box; }
  body {
    font-family: 'Inter', -apple-system, BlinkMacSystemFont, sans-serif;
    background: #0f1117;
    color: #e4e4e7;
    min-height: 100vh;
  }
  .header {
    text-align: center;
    padding: 40px 20px 12px;
  }
  .header h1 {
    font-size: 2em;
    font-weight: 800;
    letter-spacing: -0.03em;
    background: linear-gradient(135deg, #667eea, #f5576c, #a18cd1, #43e97b);
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
    background-size: 300% 300%;
    animation: gradient 8s ease infinite;
  }
  @keyframes gradient {
    0%, 100% { background-position: 0% 50%; }
    50% { background-position: 100% 50%; }
  }
  .header p { color: #71717a; margin-top: 6px; font-size: 0.95em; }
  .gpu-bar {
    max-width: 1200px;
    margin: 16px auto 24px;
    padding: 0 20px;
  }
  .gpu-info {
    background: #1a1b23;
    border: 1px solid #27272a;
    border-radius: 12px;
    padding: 16px 20px;
    font-size: 0.85em;
  }
  .gpu-top {
    display: flex; align-items: center; gap: 12px; margin-bottom: 12px;
  }
  .gpu-info .chip {
    background: #27272a; color: #a1a1aa;
    padding: 3px 10px; border-radius: 6px;
    font-weight: 600; white-space: nowrap;
  }
  .gpu-info .gpu-name { color: #e4e4e7; font-weight: 600; }
  .gpu-info .running-count { color: #22c55e; font-weight: 600; white-space: nowrap; margin-left: auto; }
  .gpu-stats {
    display: grid;
    grid-template-columns: repeat(4, 1fr);
    gap: 10px;
  }
  .gpu-stat {
    background: #0f1117;
    border: 1px solid #27272a;
    border-radius: 8px;
    padding: 10px 12px;
  }
  .gpu-stat-label {
    font-size: 0.72em; color: #52525b; font-weight: 600;
    text-transform: uppercase; letter-spacing: 0.06em;
    margin-bottom: 6px;
  }
  .gpu-stat-value {
    font-size: 1.3em; font-weight: 700; letter-spacing: -0.02em;
  }
  .gpu-stat-sub {
    font-size: 0.72em; color: #52525b; margin-top: 2px;
  }
  .gpu-bar-fill {
    height: 4px; border-radius: 2px; margin-top: 6px;
    background: #27272a; overflow: hidden;
  }
  .gpu-bar-fill-inner {
    height: 100%; border-radius: 2px;
    transition: width 0.5s ease;
  }

  .analytics-bar {
    max-width: 1200px;
    margin: 0 auto 24px;
    padding: 0 20px;
  }
  .analytics-panel {
    background: #1a1b23;
    border: 1px solid #27272a;
    border-radius: 12px;
    padding: 20px;
  }
  .analytics-top {
    display: flex; align-items: center; gap: 12px; margin-bottom: 16px; flex-wrap: wrap;
  }
  .analytics-top .chip { background: #27272a; color: #a1a1aa; padding: 3px 10px; border-radius: 6px; font-weight: 600; font-size: 0.85em; }
  .analytics-title { color: #e4e4e7; font-weight: 600; font-size: 0.85em; }
  .analytics-stats {
    display: grid;
    grid-template-columns: repeat(5, 1fr);
    gap: 10px;
    margin-bottom: 16px;
  }
  .analytics-stat {
    background: #0f1117; border: 1px solid #27272a; border-radius: 8px; padding: 10px 12px;
  }
  .analytics-stat-label { font-size: 0.72em; color: #52525b; font-weight: 600; text-transform: uppercase; letter-spacing: 0.06em; margin-bottom: 6px; }
  .analytics-stat-value { font-size: 1.15em; font-weight: 700; letter-spacing: -0.02em; }
  .analytics-stat-sub { font-size: 0.72em; color: #52525b; margin-top: 2px; }
  .analytics-charts {
    display: grid; grid-template-columns: 2fr 1fr; gap: 16px; margin-top: 4px;
  }
  .analytics-chart-box {
    background: #0f1117; border: 1px solid #27272a; border-radius: 8px; padding: 12px;
    position: relative; height: 180px;
  }
  .analytics-chart-title { font-size: 0.72em; color: #52525b; font-weight: 600; text-transform: uppercase; letter-spacing: 0.06em; margin-bottom: 10px; }
  .kwh-row {
    display: flex; align-items: center; gap: 8px; margin-left: auto;
  }
  .range-btn {
    background: #0f1117; border: 1px solid #27272a; border-radius: 6px;
    color: #71717a; padding: 3px 10px; font-size: 0.75em; cursor: pointer;
    transition: all 0.15s;
  }
  .range-btn:hover { border-color: #3f3f46; color: #e4e4e7; }
  .range-btn.active { background: #27272a; color: #e4e4e7; border-color: #3f3f46; }
  .kwh-row label { font-size: 0.75em; color: #71717a; white-space: nowrap; }
  .kwh-row input {
    background: #27272a; border: 1px solid #3f3f46; border-radius: 6px;
    color: #e4e4e7; padding: 4px 8px; font-size: 0.8em; width: 80px;
    outline: none;
  }
  .kwh-row input:focus { border-color: #667eea; }
  @media (max-width: 700px) {
    .analytics-stats { grid-template-columns: repeat(2, 1fr); }
    .analytics-charts { grid-template-columns: 1fr; }
  }

  .category-label {
    max-width: 1200px;
    margin: 24px auto 8px;
    padding: 0 20px;
    font-size: 0.8em;
    font-weight: 700;
    text-transform: uppercase;
    letter-spacing: 0.1em;
    color: #52525b;
  }
  .grid {
    max-width: 1200px;
    margin: 0 auto;
    padding: 0 20px 8px;
    display: grid;
    grid-template-columns: repeat(auto-fill, minmax(360px, 1fr));
    gap: 12px;
  }
  .card {
    background: #1a1b23;
    border: 1px solid #27272a;
    border-radius: 14px;
    padding: 20px;
    transition: border-color 0.2s;
  }
  .card:hover { border-color: #3f3f46; }
  .card-top {
    display: flex;
    align-items: flex-start;
    justify-content: space-between;
    margin-bottom: 12px;
  }
  .card-icon {
    width: 38px; height: 38px; border-radius: 10px;
    display: flex; align-items: center; justify-content: center;
    font-size: 16px; font-weight: 700;
  }
  .card h2 { font-size: 1.05em; font-weight: 700; margin-bottom: 2px; }
  .card .desc { color: #71717a; font-size: 0.82em; margin-bottom: 14px; line-height: 1.4; }
  .tag-badge {
    display: inline-block;
    background: #1e1e2e; color: #71717a;
    padding: 2px 8px; border-radius: 4px;
    font-size: 0.7em; font-weight: 500;
    border: 1px solid #27272a;
  }
  .card .port-badge {
    display: inline-block;
    background: #27272a; color: #a1a1aa;
    padding: 2px 8px; border-radius: 4px;
    font-size: 0.72em; font-weight: 600;
    font-family: monospace;
    margin-bottom: 12px;
  }
  .status-row {
    display: flex; align-items: center; gap: 8px; margin-bottom: 12px;
  }
  .status-dot { width: 9px; height: 9px; border-radius: 50%; flex-shrink: 0; }
  .status-dot.stopped { background: #52525b; }
  .status-dot.starting { background: #facc15; animation: pulse 1.5s infinite; }
  .status-dot.ready { background: #22c55e; box-shadow: 0 0 8px #22c55e55; }
  @keyframes pulse { 0%, 100% { opacity: 1; } 50% { opacity: 0.3; } }
  .status-text {
    font-size: 0.78em; color: #a1a1aa; font-weight: 600;
    text-transform: uppercase; letter-spacing: 0.06em;
  }
  .card-actions { display: flex; gap: 8px; }
  .btn {
    flex: 1; padding: 9px 14px; border: none; border-radius: 8px;
    font-weight: 600; font-size: 0.85em; cursor: pointer;
    transition: opacity 0.15s, transform 0.1s;
    display: flex; align-items: center; justify-content: center; gap: 5px;
    text-decoration: none;
  }
  .btn:hover { opacity: 0.85; }
  .btn:active { transform: scale(0.97); }
  .btn-start { background: #22c55e; color: #fff; }
  .btn-stop { background: #ef4444; color: #fff; }
  .btn-open { background: transparent; border: 1px solid #3f3f46; color: #e4e4e7; }
  .btn-open:hover { background: #27272a; }
  .btn:disabled { opacity: 0.35; cursor: not-allowed; transform: none; }
  .offload-row {
    display: flex; align-items: center; gap: 8px; margin-bottom: 10px;
    font-size: 0.78em; color: #a1a1aa;
  }
  .offload-row label { cursor: pointer; user-select: none; display: flex; align-items: center; gap: 6px; }
  .offload-toggle {
    position: relative; width: 36px; height: 20px; appearance: none;
    background: #27272a; border-radius: 10px; cursor: pointer; transition: background 0.2s;
    outline: none; border: 1px solid #3f3f46; flex-shrink: 0;
  }
  .offload-toggle:checked { background: #3b82f6; border-color: #3b82f6; }
  .offload-toggle::after {
    content: ''; position: absolute; top: 2px; left: 2px;
    width: 14px; height: 14px; border-radius: 50%;
    background: #e4e4e7; transition: transform 0.2s;
  }
  .offload-toggle:checked::after { transform: translateX(16px); }
  .log-toggle {
    margin-top: 10px; font-size: 0.75em; color: #52525b;
    cursor: pointer; user-select: none;
  }
  .log-toggle:hover { color: #a1a1aa; }
  .log-box {
    margin-top: 6px; background: #0f1117; border: 1px solid #27272a;
    border-radius: 6px; padding: 10px;
    font-family: 'JetBrains Mono', 'Fira Code', monospace;
    font-size: 0.7em; color: #71717a;
    max-height: 180px; overflow-y: auto;
    white-space: pre-wrap; word-break: break-all;
    display: none;
  }
  .log-box.open { display: block; }
  .filter-bar {
    max-width: 1200px;
    margin: 0 auto 16px;
    padding: 0 20px;
    display: flex;
    flex-wrap: wrap;
    gap: 8px;
    align-items: center;
  }
  .filter-label {
    font-size: 0.78em; font-weight: 600; color: #52525b;
    text-transform: uppercase; letter-spacing: 0.06em;
    margin-right: 4px;
  }
  .filter-chip {
    padding: 5px 14px; border-radius: 20px;
    font-size: 0.78em; font-weight: 600;
    border: 1px solid #27272a; background: transparent; color: #a1a1aa;
    cursor: pointer; transition: all 0.15s; user-select: none;
  }
  .filter-chip:hover { border-color: #3f3f46; color: #e4e4e7; }
  .filter-chip.active { background: #27272a; color: #e4e4e7; border-color: #3f3f46; }
  .card.hidden { display: none; }
  .category-section.hidden { display: none; }
  .footer {
    text-align: center; padding: 24px; color: #3f3f46; font-size: 0.75em;
  }
</style>
</head>
<body>

<div class="header">
  <h1>GPU Model Dashboard</h1>
  <p>NVIDIA RTX PRO 6000 &middot; 96 GB VRAM</p>
</div>

<div class="gpu-bar">
  <div class="gpu-info">
    <div class="gpu-top">
      <span class="chip">GPU</span>
      <span class="gpu-name" id="gpu-name">...</span>
      <span class="running-count" id="running-count">0 running</span>
    </div>
    <div class="gpu-stats">
      <div class="gpu-stat">
        <div class="gpu-stat-label">VRAM</div>
        <div class="gpu-stat-value" id="gpu-vram">--</div>
        <div class="gpu-stat-sub" id="gpu-vram-sub"></div>
        <div class="gpu-bar-fill"><div class="gpu-bar-fill-inner" id="gpu-vram-bar" style="width:0%;background:#667eea"></div></div>
      </div>
      <div class="gpu-stat">
        <div class="gpu-stat-label">Core Usage</div>
        <div class="gpu-stat-value" id="gpu-util">--</div>
        <div class="gpu-stat-sub">&nbsp;</div>
        <div class="gpu-bar-fill"><div class="gpu-bar-fill-inner" id="gpu-util-bar" style="width:0%;background:#22c55e"></div></div>
      </div>
      <div class="gpu-stat">
        <div class="gpu-stat-label">Temperature</div>
        <div class="gpu-stat-value" id="gpu-temp">--</div>
        <div class="gpu-stat-sub" id="gpu-temp-sub">&nbsp;</div>
        <div class="gpu-bar-fill"><div class="gpu-bar-fill-inner" id="gpu-temp-bar" style="width:0%;background:#facc15"></div></div>
      </div>
      <div class="gpu-stat">
        <div class="gpu-stat-label">Power</div>
        <div class="gpu-stat-value" id="gpu-power">--</div>
        <div class="gpu-stat-sub" id="gpu-power-sub"></div>
        <div class="gpu-bar-fill"><div class="gpu-bar-fill-inner" id="gpu-power-bar" style="width:0%;background:#f5576c"></div></div>
      </div>
    </div>
  </div>
</div>

<div class="analytics-bar">
  <div class="analytics-panel">
    <div class="analytics-top">
      <span class="chip">OPENCODE</span>
      <span class="analytics-title">Token Usage &amp; Power Cost</span>
      <div style="display:flex;gap:4px;margin-left:12px">
        <button class="range-btn active" onclick="setRange(7)">7d</button>
        <button class="range-btn" onclick="setRange(30)">30d</button>
        <button class="range-btn" onclick="setRange(90)">90d</button>
        <button class="range-btn" onclick="setRange(365)">1y</button>
        <button class="range-btn" onclick="setRange(0)">All</button>
      </div>
      <div class="kwh-row" style="margin-left:auto">
        <label>$/kWh</label>
        <input type="number" id="kwh-rate" value="0.12" step="0.01" min="0" onchange="updateCost(); updateDailyChart()">
      </div>
    </div>
    <div class="analytics-stats">
      <div class="analytics-stat">
        <div class="analytics-stat-label">Total Tokens</div>
        <div class="analytics-stat-value" id="stat-total">--</div>
        <div class="analytics-stat-sub" id="stat-range-label">last 7 days</div>
      </div>
      <div class="analytics-stat">
        <div class="analytics-stat-label">Input Tokens</div>
        <div class="analytics-stat-value" id="stat-input">--</div>
        <div class="analytics-stat-sub" id="stat-input-label">last 7 days</div>
      </div>
      <div class="analytics-stat">
        <div class="analytics-stat-label">Output Tokens</div>
        <div class="analytics-stat-value" id="stat-output">--</div>
        <div class="analytics-stat-sub" id="stat-output-label">last 7 days</div>
      </div>
      <div class="analytics-stat">
        <div class="analytics-stat-label">Power Cost/hr</div>
        <div class="analytics-stat-value" id="stat-cost-hr">--</div>
        <div class="analytics-stat-sub" id="stat-cost-sub">at current draw</div>
      </div>
      <div class="analytics-stat">
        <div class="analytics-stat-label">Est. Today Cost</div>
        <div class="analytics-stat-value" id="stat-cost-day">--</div>
        <div class="analytics-stat-sub">power only</div>
      </div>
      <div class="analytics-stat">
        <div class="analytics-stat-label">Total Cost (Range)</div>
        <div class="analytics-stat-value" id="stat-cost-total">--</div>
        <div class="analytics-stat-sub" id="stat-cost-total-sub">power only, includes backfill</div>
      </div>
    </div>
    <div class="analytics-charts">
      <div class="analytics-chart-box">
        <div style="display:flex;align-items:center;justify-content:space-between;margin-bottom:10px">
          <div class="analytics-chart-title" style="margin-bottom:0" id="chart-daily-title">Tokens per Day</div>
          <div style="display:flex;gap:3px">
            <button class="range-btn active" id="toggle-tokens" onclick="setChartMode('tokens')">Tokens</button>
            <button class="range-btn" id="toggle-cost" onclick="setChartMode('cost')">$ Cost</button>
            <button class="range-btn" id="toggle-power" onclick="setChartMode('power')">Power (kWh)</button>
          </div>
        </div>
        <canvas id="chart-daily"></canvas>
      </div>
      <div class="analytics-chart-box">
        <div class="analytics-chart-title">Top Models by Tokens</div>
        <canvas id="chart-models"></canvas>
      </div>
    </div>
  </div>
</div>

<div class="filter-bar" id="filter-bar">
  <span class="filter-label">Filter</span>
  <button class="filter-chip active" data-filter="all" onclick="setFilter('all')">All</button>
</div>

<div id="content"></div>

<div class="footer">Load &middot; Use &middot; Unload</div>

<script src="https://cdn.jsdelivr.net/npm/chart.js@4.4.0/dist/chart.umd.min.js"></script>
<script>
const HOST = location.hostname;
const CATEGORY_ORDER = ['Image', 'Video', 'Audio', 'LLM', 'Tools'];
let currentModels = {};
let activeFilter = 'all';
let filtersBuilt = false;

function buildFilters(models) {
  if (filtersBuilt) return;
  const tagSet = new Set();
  for (const m of Object.values(models)) {
    (m.tags || []).forEach(t => tagSet.add(t));
  }
  const bar = document.getElementById('filter-bar');
  const sorted = [...tagSet].sort();
  for (const tag of sorted) {
    const btn = document.createElement('button');
    btn.className = 'filter-chip';
    btn.dataset.filter = tag;
    btn.textContent = tag;
    btn.onclick = () => setFilter(tag);
    bar.appendChild(btn);
  }
  // Add "running" filter
  const runBtn = document.createElement('button');
  runBtn.className = 'filter-chip';
  runBtn.dataset.filter = '_running';
  runBtn.textContent = 'running';
  runBtn.onclick = () => setFilter('_running');
  bar.appendChild(runBtn);
  filtersBuilt = true;
}

function setFilter(f) {
  activeFilter = f;
  document.querySelectorAll('.filter-chip').forEach(c => {
    c.classList.toggle('active', c.dataset.filter === f);
  });
  applyFilter();
}

function applyFilter() {
  for (const [id, m] of Object.entries(currentModels)) {
    const card = document.getElementById('card-' + id);
    if (!card) continue;
    let show = true;
    if (activeFilter === '_running') {
      show = m.status !== 'stopped';
    } else if (activeFilter !== 'all') {
      show = (m.tags || []).includes(activeFilter);
    }
    card.classList.toggle('hidden', !show);
  }
  // Hide empty category sections
  for (const cat of CATEGORY_ORDER) {
    const label = document.getElementById('label-' + cat);
    const grid = document.getElementById('grid-' + cat);
    if (!label || !grid) continue;
    const visibleCards = grid.querySelectorAll('.card:not(.hidden)');
    const hide = visibleCards.length === 0;
    label.classList.toggle('hidden', hide);
    grid.classList.toggle('hidden', hide);
  }
}

function renderAll(models) {
  currentModels = models;
  buildFilters(models);
  const content = document.getElementById('content');
  let runCount = 0;

  // Group by category
  const groups = {};
  for (const [id, m] of Object.entries(models)) {
    if (!groups[m.category]) groups[m.category] = [];
    groups[m.category].push([id, m]);
    if (m.status !== 'stopped') runCount++;
  }

  document.getElementById('running-count').textContent = runCount + ' model' + (runCount !== 1 ? 's' : '') + ' running';

  // Build HTML if first render
  if (!content.dataset.rendered) {
    let html = '';
    for (const cat of CATEGORY_ORDER) {
      if (!groups[cat]) continue;
      html += `<div class="category-label" id="label-${cat}">${cat}</div><div class="grid" id="grid-${cat}">`;
      for (const [id, m] of groups[cat]) {
        const tagBadges = (m.tags || []).map(t => `<span class="tag-badge">${t}</span>`).join('');
        html += `
        <div class="card" id="card-${id}" data-tags="${(m.tags||[]).join(',')}">
          <div class="card-top">
            <div>
              <h2>${m.name}</h2>
              <div class="desc">${m.description}</div>
              <div style="display:flex;gap:6px;flex-wrap:wrap;align-items:center;margin-bottom:12px">
                <span class="port-badge">:${m.port}</span>
                ${tagBadges}
              </div>
            </div>
            <div class="card-icon" style="background:${m.color}18;color:${m.color}">&bull;</div>
          </div>
          <div class="status-row">
            <div class="status-dot" id="dot-${id}"></div>
            <span class="status-text" id="status-${id}"></span>
          </div>
          ${m.supports_offload ? `<div class="offload-row" id="offload-row-${id}"><label><input type="checkbox" class="offload-toggle" id="offload-${id}"> CPU Offload</label></div>` : ''}
          <div class="card-actions" id="actions-${id}"></div>
          <div class="log-toggle" onclick="toggleLog('${id}')">view logs</div>
          <div class="log-box" id="log-${id}"></div>
        </div>`;
      }
      html += '</div>';
    }
    content.innerHTML = html;
    content.dataset.rendered = '1';
  }

  // Update statuses
  for (const [id, m] of Object.entries(models)) {
    const dot = document.getElementById('dot-' + id);
    const st = document.getElementById('status-' + id);
    const actions = document.getElementById('actions-' + id);
    if (!dot) continue;

    dot.className = 'status-dot ' + m.status;
    st.textContent = m.status === 'ready' ? 'Running' : m.status === 'starting' ? 'Starting...' : 'Stopped';

    const offloadRow = document.getElementById('offload-row-' + id);
    if (offloadRow) offloadRow.style.display = m.status === 'stopped' ? 'flex' : 'none';

    const proto = m.protocol || 'http';
    if (m.status === 'stopped') {
      actions.innerHTML = `<button class="btn btn-start" onclick="startModel('${id}')">Start</button>`;
    } else if (m.status === 'starting') {
      actions.innerHTML = `<button class="btn btn-stop" onclick="stopModel('${id}')">Stop</button>
        <button class="btn btn-open" disabled>Starting...</button>`;
    } else {
      actions.innerHTML = `<button class="btn btn-stop" onclick="stopModel('${id}')">Stop</button>
        <a class="btn btn-open" href="${proto}://${HOST}:${m.port}${m.path||''}" target="_blank" rel="noopener">Open</a>`;
    }
  }
  applyFilter();
}

async function refresh() {
  try {
    const r = await fetch('/api/status');
    renderAll(await r.json());
  } catch (e) {}
}

async function startModel(id) {
  const btn = event.target;
  btn.disabled = true; btn.textContent = 'Starting...';
  const cb = document.getElementById('offload-' + id);
  const offload = cb && cb.checked ? 'true' : 'false';
  await fetch('/api/start/' + id + '?offload=' + offload, { method: 'POST' });
  setTimeout(refresh, 500);
}

async function stopModel(id) {
  const btn = event.target;
  btn.disabled = true; btn.textContent = 'Stopping...';
  await fetch('/api/stop/' + id, { method: 'POST' });
  setTimeout(refresh, 500);
}

const logIntervals = {};
async function toggleLog(id) {
  const box = document.getElementById('log-' + id);
  if (box.classList.contains('open')) {
    box.classList.remove('open');
    if (logIntervals[id]) { clearInterval(logIntervals[id]); delete logIntervals[id]; }
    return;
  }
  async function fetchLog() {
    try {
      const r = await fetch('/api/logs/' + id);
      const data = await r.json();
      const wasAtBottom = box.scrollHeight - box.scrollTop - box.clientHeight < 40;
      box.textContent = data.logs || '(no logs yet)';
      if (wasAtBottom) box.scrollTop = box.scrollHeight;
    } catch (e) { box.textContent = '(error)'; }
  }
  await fetchLog();
  box.classList.add('open');
  box.scrollTop = box.scrollHeight;
  logIntervals[id] = setInterval(fetchLog, 3000);
}

// ── Token analytics ──────────────────────────────────────────────
let tokenData = null;
let powerByDay = {};
let powerMeta = {};
let currentPowerW = 0;
let todayWh = 0;
let chartDaily = null;
let chartModels = null;
let currentRange = 7;
let chartMode = 'tokens';

function fmtNum(n) {
  if (n >= 1e9) return (n/1e9).toFixed(2) + 'B';
  if (n >= 1e6) return (n/1e6).toFixed(1) + 'M';
  if (n >= 1e3) return (n/1e3).toFixed(1) + 'K';
  return n.toString();
}

function setRange(days) {
  currentRange = days;
  document.querySelectorAll('.range-btn').forEach(b => {
    const label = b.textContent.trim();
    const match = days === 7 && label === '7d' || days === 30 && label === '30d' ||
                  days === 90 && label === '90d' || days === 365 && label === '1y' ||
                  days === 0 && label === 'All';
    b.classList.toggle('active', match);
  });
  const rangeLabel = days === 0 ? 'all time' : 'last ' + days + ' days';
  document.getElementById('stat-range-label').textContent = rangeLabel;
  document.getElementById('stat-input-label').textContent = rangeLabel;
  document.getElementById('stat-output-label').textContent = rangeLabel;
  refreshTokens();
  // Re-fetch power data to ensure alignment with new range
  refreshPower();
}

function updateCost() {
  const rate = parseFloat(document.getElementById('kwh-rate').value) || 0.12;
  if (!currentPowerW) return;
  // Cost/hr from current draw
  const costHr = (currentPowerW / 1000) * rate;
  document.getElementById('stat-cost-hr').textContent = '$' + costHr.toFixed(3);
  document.getElementById('stat-cost-sub').textContent = currentPowerW.toFixed(0) + 'W @ $' + rate + '/kWh';
  // Today cost from actual accumulated watt-hours
  const costToday = (todayWh / 1000) * rate;
  document.getElementById('stat-cost-day').textContent = '$' + costToday.toFixed(3);
}

async function refreshPower() {
  try {
    const r = await fetch('/api/power');
    const d = await r.json();
    todayWh = d.today_wh || 0;
    powerByDay = d.by_day || {};
    powerMeta = { has_backfill: d.has_backfill || false, backfill_days: d.backfill_days || 0, real_days: d.real_days || 0 };
    updateCost();
    if (tokenData) {
      updateTotalCost();
      updateDailyChart();
    }
  } catch(e) {}
}

function updateTotalCost() {
  const rate = parseFloat(document.getElementById('kwh-rate').value) || 0.12;
  if (!tokenData) return;
  
  // Calculate total cost for the current range
  // Only sum power for days that exist in the token data (i.e., days with actual usage)
  let totalWh = 0;
  const days = Object.keys(tokenData.by_day || {});
  for (const day of days) {
    totalWh += powerByDay[day] || 0;
  }
  
  const totalCost = (totalWh / 1000) * rate;
  document.getElementById('stat-cost-total').textContent = '$' + totalCost.toFixed(2);
  
  // Show range info
  let rangeInfo = '';
  if (currentRange === 0) {
    rangeInfo = 'all time';
  } else {
    rangeInfo = 'last ' + currentRange + ' days';
  }
  
  let subParts = ['power only', rangeInfo];
  if (powerMeta.has_backfill) {
    subParts.push(powerMeta.backfill_days + 'd backfilled, ' + powerMeta.real_days + 'd real');
  }
  document.getElementById('stat-cost-total-sub').textContent = subParts.join(' · ');
}

// Helper to check if a day is in real power data (before backfill)
function _loadPowerLog() {
  // powerMeta stores real_days count, but we don't have the actual set
  // For the sub-label, we just use the metadata counts
  return { _real_days: powerMeta.real_days || 0 };
}

function setChartMode(mode) {
  chartMode = mode;
  document.getElementById('toggle-tokens').classList.toggle('active', mode === 'tokens');
  document.getElementById('toggle-cost').classList.toggle('active', mode === 'cost');
  document.getElementById('toggle-power').classList.toggle('active', mode === 'power');
  updateDailyChart();
}

function updateDailyChart() {
  if (!chartDaily || !tokenData) return;
  const rate = parseFloat(document.getElementById('kwh-rate').value) || 0.12;
  const days = Object.keys(tokenData.by_day);
  const shortDays = days.map(d => d.slice(5));

  if (chartMode === 'tokens') {
    document.getElementById('chart-daily-title').textContent = 'Tokens per Day';
    chartDaily.data.labels = shortDays;
    chartDaily.data.datasets = [
      { label: 'Input', data: days.map(k => tokenData.by_day[k].input), backgroundColor: 'rgba(102,126,234,0.7)', stack: 's' },
      { label: 'Output', data: days.map(k => tokenData.by_day[k].output), backgroundColor: 'rgba(34,197,94,0.7)', stack: 's' },
    ];
    chartDaily.options.scales.y.ticks.callback = v => fmtNum(v);
  } else if (chartMode === 'cost') {
    document.getElementById('chart-daily-title').textContent = 'Power Cost per Day ($)';
    chartDaily.data.labels = shortDays;
    chartDaily.data.datasets = [
      { label: 'Cost ($)', data: days.map(k => {
          const wh = powerByDay[k] || 0;
          return +((wh / 1000) * rate).toFixed(4);
        }),
        backgroundColor: 'rgba(249,115,22,0.7)', stack: 's'
      },
    ];
    chartDaily.options.scales.y.ticks.callback = v => '$' + v.toFixed(3);
  } else {
    document.getElementById('chart-daily-title').textContent = 'Power Consumption per Day (kWh)';
    chartDaily.data.labels = shortDays;
    chartDaily.data.datasets = [
      { label: 'Power (kWh)', data: days.map(k => {
          const wh = powerByDay[k] || 0;
          return +(wh / 1000).toFixed(3);
        }),
        backgroundColor: 'rgba(168,85,247,0.7)', stack: 's'
      },
    ];
    chartDaily.options.scales.y.ticks.callback = v => v.toFixed(3) + ' kWh';
  }
  chartDaily.update();
}

async function refreshTokens() {
  try {
    const url = '/api/tokens?days=' + currentRange;
    const r = await fetch(url);
    const d = await r.json();
    if (d.error) return;
    tokenData = d;
    // Sum totals from the filtered day range (not all-time)
    let rangeInput = 0, rangeOutput = 0, rangeTotal = 0;
    for (const v of Object.values(d.by_day)) {
      rangeInput += v.input;
      rangeOutput += v.output;
      rangeTotal += v.total;
    }
    document.getElementById('stat-total').textContent = fmtNum(rangeTotal);
    document.getElementById('stat-input').textContent = fmtNum(rangeInput);
    document.getElementById('stat-output').textContent = fmtNum(rangeOutput);

    // Daily chart
    const days = Object.keys(d.by_day);
    const shortDays = days.map(k => k.slice(5));
    if (!chartDaily) {
      const ctx = document.getElementById('chart-daily').getContext('2d');
      chartDaily = new Chart(ctx, {
        type: 'bar',
        data: { labels: shortDays, datasets: [] },
        options: {
          responsive: true, maintainAspectRatio: false,
          plugins: { legend: { labels: { color: '#71717a', font: { size: 10 } } } },
          scales: {
            x: { stacked: true, ticks: { color: '#52525b', font: { size: 9 }, maxRotation: 45 }, grid: { color: '#1f2028' } },
            y: { stacked: true, ticks: { color: '#52525b', font: { size: 9 }, callback: v => fmtNum(v) }, grid: { color: '#1f2028' } }
          }
        }
      });
    }
    updateDailyChart();
    if (powerByDay && Object.keys(powerByDay).length > 0) {
      updateTotalCost();
    }

    // Model chart
    const models = Object.keys(d.by_model);
    const modelTotals = models.map(k => d.by_model[k].total);
    const shortModels = models.map(m => m.length > 22 ? m.slice(0, 22) + '…' : m);
    const colors = ['#667eea','#22c55e','#f5576c','#facc15','#a78bfa','#06b6d4','#f97316','#ec4899','#84cc16','#14b8a6'];

    if (!chartModels) {
      const ctx2 = document.getElementById('chart-models').getContext('2d');
      chartModels = new Chart(ctx2, {
        type: 'doughnut',
        data: {
          labels: shortModels,
          datasets: [{ data: modelTotals, backgroundColor: colors, borderWidth: 0 }]
        },
        options: {
          responsive: true, maintainAspectRatio: false,
          plugins: {
            legend: { position: 'right', labels: { color: '#71717a', font: { size: 9 }, boxWidth: 10 } },
            tooltip: { callbacks: { label: ctx => ctx.label + ': ' + fmtNum(ctx.raw) } }
          }
        }
      });
    } else {
      chartModels.data.labels = shortModels;
      chartModels.data.datasets[0].data = modelTotals;
      chartModels.update();
    }
  } catch(e) {}
}

// Hook into GPU refresh to update cost
const _origRefreshGpu = refreshGpu;
async function refreshGpu() {
  try {
    const r = await fetch('/api/gpu');
    const g = await r.json();
    if (g.error) return;
    currentPowerW = g.power;
    const vramPct = (g.vram_used / g.vram_total * 100);
    const vramGB = (g.vram_used / 1024).toFixed(1);
    const vramTotalGB = (g.vram_total / 1024).toFixed(0);
    document.getElementById('gpu-name').textContent = g.name;
    document.getElementById('gpu-vram').textContent = vramGB + ' GB';
    document.getElementById('gpu-vram-sub').textContent = 'of ' + vramTotalGB + ' GB (' + vramPct.toFixed(0) + '%)';
    document.getElementById('gpu-vram-bar').style.width = vramPct + '%';
    document.getElementById('gpu-util').textContent = g.gpu_util + '%';
    document.getElementById('gpu-util-bar').style.width = g.gpu_util + '%';
    const tempColor = g.temp > 80 ? '#ef4444' : g.temp > 65 ? '#facc15' : '#22c55e';
    document.getElementById('gpu-temp').textContent = g.temp + '\u00b0C';
    document.getElementById('gpu-temp-bar').style.width = (g.temp / 100 * 100) + '%';
    document.getElementById('gpu-temp-bar').style.background = tempColor;
    document.getElementById('gpu-power').textContent = g.power.toFixed(0) + ' W';
    document.getElementById('gpu-power-sub').textContent = 'of ' + g.power_limit.toFixed(0) + ' W';
    document.getElementById('gpu-power-bar').style.width = (g.power / g.power_limit * 100) + '%';
    updateCost();
  } catch(e) {}
}

refresh();
refreshGpu();
refreshTokens();
refreshPower();
setInterval(refresh, 5000);
setInterval(refreshGpu, 3000);
setInterval(refreshTokens, 60000);
setInterval(refreshPower, 60000);
</script>
</body>
</html>
"""

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=80)
