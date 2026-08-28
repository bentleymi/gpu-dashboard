import subprocess
import signal
import socket
import os
import re
import threading
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
        "cwd": "/mnt/raid1_nvme/JanusPro7b",
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
        "cmd": ["/mnt/raid1_nvme/JanusPro7b/flux-venv/bin/python", "flux_app.py"],
        "cwd": "/mnt/raid1_nvme/JanusPro7b",
        "env": {},
        "protocol": "https",
        "category": "Image",
        "icon": "image",
        "color": "#f5576c",
        "tags": ["text-to-image", "image-to-image"],
        "supports_offload": True,
    },
    "flux_uncensored": {
        "name": "FLUX Klein 9B Uncensored",
        "description": "Uncensored image generation with Lustly.ai LoRA — unrestricted content",
        "port": 4243,
        "cmd": ["/mnt/raid1_nvme/JanusPro7b/flux-venv/bin/python", "flux_uncensored_app.py"],
        "cwd": "/mnt/raid1_nvme/JanusPro7b",
        "env": {},
        "protocol": "https",
        "category": "Image",
        "icon": "image",
        "color": "#e74c3c",
        "tags": ["text-to-image", "image-to-image", "uncensored"],
        "supports_offload": True,
    },
    "qwen_edit_general": {
        "name": "Qwen Image Edit (General)",
        "description": "General purpose image editing: upload + prompt → edited result (Qwen/Qwen-Image-Edit-2511)",
        "port": 4041,
        "cmd": ["/mnt/raid1_nvme/JanusPro7b/qwen-venv/bin/python", "qwen_edit_general.py"],
        "cwd": "/mnt/raid1_nvme/JanusPro7b",
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
        "cmd": ["/mnt/raid1_nvme/JanusPro7b/qwen-venv/bin/python", "qwen_edit_dual.py"],
        "cwd": "/mnt/raid1_nvme/JanusPro7b",
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
        "cmd": ["/mnt/raid1_nvme/JanusPro7b/qwen-venv/bin/python", "qwen_app.py"],
        "cwd": "/mnt/raid1_nvme/JanusPro7b",
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
    "flux2_uncensored": {
        "name": "FLUX.2 Klein Uncensored 9B",
        "description": "FLUX.2 Klein base 9B uncensored - fast image generation",
        "port": 9002,
        "cmd": ["/mnt/raid1_nvme/JanusPro7b/flux2-uncensored-venv/bin/python", "flux2_uncensored_app.py"],
        "cwd": "/mnt/raid1_nvme/JanusPro7b",
        "env": {},
        "protocol": "https",
        "category": "Image",
        "icon": "image",
        "color": "#ec4899",
        "tags": ["text-to-image", "uncensored"],
        "supports_offload": True,
    },
    # --- Video Generation ---
    "ltx": {
        "name": "LTX-2 Video",
        "description": "Audio-video generation from text and images (19B)",
        "port": 4141,
        "cmd": ["/mnt/raid1_nvme/JanusPro7b/ltx-venv/bin/python", "ltx_app.py"],
        "cwd": "/mnt/raid1_nvme/JanusPro7b",
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
    "wan_uncensored": {
        "name": "WAN 2.2 NSFW LoRAs",
        "description": "Uncensored image-to-video with specialized NSFW LoRAs",
        "port": 4245,
        "cmd": ["/mnt/raid1_sata/wan2.2-TI2V-5B/venv/bin/python", "wan2.2_nsfw_loras_app.py"],
        "cwd": "/mnt/raid1_nvme/JanusPro7b",
        "env": {},
        "protocol": "https",
        "category": "Video",
        "icon": "clapperboard",
        "color": "#e67e22",
        "tags": ["text-to-video", "image-to-video", "uncensored"],
        "supports_offload": True,
    },
    "ltx_nsFW": {
        "name": "LTX 2.3 NSFW Motion",
        "description": "NSFW motion LoRA for LTX video generation — stacked LoRAs",
        "port": 4246,
        "cmd": ["/mnt/raid1_nvme/JanusPro7b/ltx-venv/bin/python", "ltx_nsfw_lora_app.py"],
        "cwd": "/mnt/raid1_nvme/JanusPro7b",
        "env": {"PYTORCH_CUDA_ALLOC_CONF": "expandable_segments:True", "CPU_OFFLOAD": "1", "CUDA_LAUNCH_BLOCKING": "0"},
        "protocol": "https",
        "category": "Video",
        "icon": "film",
        "color": "#8e44ad",
        "tags": ["text-to-video", "image-to-video", "nsfw"],
        "supports_offload": True,
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
        "port": 8093,
        "systemd_service": "rvc-audio.service",
        "cmd": ["/mnt/raid1_nvme/vcs-audio-5.0.0/venv/bin/python", "-m", "src.api.main"],
        "cwd": "/mnt/raid1_nvme/vcs-audio-5.0.0",
        "env": {"SERVER_PORT": "8093"},
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
        "systemd_service": "voice-llm.service",
        "cmd": ["/mnt/raid1_nvme/voice/bin/python3", "server.py"],
        "cwd": "/mnt/raid1_nvme/voice/higgs-audio",
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
    "minimax_music3": {
        "name": "MiniMax Music 3",
        "description": "Lyrics + caption → 32kHz stereo songs up to 5 min — 8B Qwen3 backbone + flow-matching DIT (BF16, SGLang-Omni, /v1/audio/speech API)",
        "port": 7873,
        "cmd": ["/mnt/raid1_nvme/models/MiniMax-Music3/venv/bin/sgl-omni",
                "serve", "--model-path", "/mnt/raid1_nvme/models/MiniMax-Music3",
                "--host", "0.0.0.0", "--port", "7873"],
        "cwd": "/mnt/raid1_nvme/models/MiniMax-Music3",
        "env": {"CUDA_VISIBLE_DEVICES": "0"},
        "protocol": "http",
        "category": "Audio",
        "icon": "music",
        "color": "#ff8c42",
        "tags": ["text-to-music", "music", "lyrics"],
        "supports_offload": False,
        "vram_gb": 70,
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
        "cwd": "/mnt/raid1_nvme/vllm-servers",
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
        "cwd": "/mnt/raid1_nvme/vllm-servers",
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
        "cwd": "/mnt/raid1_nvme/vllm-servers",
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
        "cwd": "/mnt/raid1_nvme/vllm-servers",
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
        "cwd": "/mnt/raid1_nvme/vllm-servers",
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
        "cwd": "/mnt/raid1_nvme/JanusPro7b",
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
        "cwd": "/mnt/raid1_nvme/JanusPro7b",
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
        "cwd": "/mnt/raid1_nvme/vllm-servers",
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
        "cwd": "/mnt/raid1_nvme/JanusPro7b",
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
        "cmd": ["/mnt/raid1_sata/vllm-env/bin/vllm", "serve", "/mnt/raid1_nvme/models/qwen36-27b",
                "--port", "8006", "--host", "0.0.0.0",
                "--enable-auto-tool-choice", "--tool-call-parser", "hermes",
                "--tensor-parallel-size", "1", "--gpu-memory-utilization", "0.90",
                "--max-model-len", "131072", "--trust-remote-code"],
        "env": {"PYTORCH_ALLOC_CONF": "expandable_segments:True"},
        "cwd": "/mnt/raid1_nvme/vllm-servers",
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
        "cmd": ["/mnt/raid1_sata/vllm-env/bin/vllm", "serve", "/mnt/raid1_nvme/models/qwen36-35b",
                "--port", "8012", "--host", "0.0.0.0",
                "--enable-auto-tool-choice", "--tool-call-parser", "hermes",
                "--tensor-parallel-size", "1", "--gpu-memory-utilization", "0.92",
                "--max-model-len", "131072", "--trust-remote-code",
                "--enforce-eager"],
        "env": {"PYTORCH_ALLOC_CONF": "expandable_segments:True"},
        "cwd": "/mnt/raid1_nvme/vllm-servers",
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
        "cmd": ["/mnt/raid1_nvme/models/ik_llama.cpp/build/bin/llama-server",
                "--model", "/mnt/raid1_nvme/models/qwen3-coder-next/Qwen3-Coder-Next-Q4_K_M/Qwen3-Coder-Next-Q4_K_M-00001-of-00004.gguf",
                "--alias", "qwen3-coder-next",
                "--ctx-size", "131072",
                "-ngl", "99",
                "-b", "2048", "-ub", "2048",
                "--host", "0.0.0.0",
                "--port", "8085",
                "--jinja",
                "--threads", "16",
                "--reasoning-tokens", "none"],
        "cwd": "/mnt/raid1_nvme/models/ik_llama.cpp",
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
        "cmd": ["/mnt/raid1_nvme/models/ik_llama.cpp/build/bin/llama-server",
                "--model", "/mnt/raid1_nvme/models/qwen3-coder-next/Qwen3-Coder-Next-Q6_K/Qwen3-Coder-Next-Q6_K-00001-of-00004.gguf",
                "--alias", "qwen3-coder-next",
                "--ctx-size", "131072",
                "-ngl", "99",
                "-b", "2048", "-ub", "2048",
                "--host", "0.0.0.0",
                "--port", "8084",
                "--jinja",
                "--threads", "16",
                "--reasoning-tokens", "none"],
        "cwd": "/mnt/raid1_nvme/models/ik_llama.cpp",
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
        "cmd": ["/mnt/raid1_nvme/models/ik_llama.cpp/build/bin/llama-server",
                "--model", "/models/qwen36-35b-uncensored/Qwen3.6-35B-A3B-Uncensored-HauhauCS-Aggressive-Q8_K_P.gguf",
                "--alias", "qwen36-35b-uncensored",
                "--ctx-size", "262144",
                "-ngl", "99",
                "-b", "2048", "-ub", "2048",
                "--cache-ram", "0",
                "--host", "0.0.0.0",
                "--port", "8086",
                "--jinja",
                "--chat-template-kwargs", '{"enable_thinking":false}',
                "--threads", "16",
                "--chunk", "4096"],
        "cwd": "/mnt/raid1_nvme/models/ik_llama.cpp",
        "env": {},
        "protocol": "http",
        "category": "LLM",
        "icon": "zap",
        "color": "#f97316",
        "tags": ["text-to-text", "text-to-code", "agents"],
        "supports_offload": False,
    },
    "qwen36_35b_uncensored_512k": {
        "name": "Qwen3.6 35B Uncensored (Q8, 512K ctx)",
        "description": "llama.cpp — Q8_K_P, 43GB, Qwen3.6 35B-A3B uncensored, YaRN-extended to 512K context",
        "port": 8097,
        "systemd_service": "qwen36-35b-uncensored-512k.service",
        "cmd": ["/mnt/raid1_nvme/models/ik_llama.cpp/build/bin/llama-server",
                "--model", "/mnt/raid1_nvme/models/qwen36-35b-uncensored/Qwen3.6-35B-A3B-Uncensored-HauhauCS-Aggressive-Q8_K_P.gguf",
                "--alias", "qwen36-35b-uncensored-512k",
                "--ctx-size", "524288",
                "--rope-scaling", "yarn",
                "--yarn-orig-ctx", "262144",
                "-ngl", "99",
                "-b", "2048", "-ub", "2048",
                "--cache-ram", "0",
                "--host", "0.0.0.0",
                "--port", "8097",
                "--jinja",
                "--chat-template-kwargs", '{"enable_thinking":false}',
                "--threads", "16",
                "--chunk", "4096"],
        "cwd": "/mnt/raid1_nvme/models/ik_llama.cpp",
        "env": {},
        "protocol": "http",
        "category": "LLM",
        "icon": "zap",
        "color": "#f97316",
        "tags": ["text-to-text", "text-to-code", "agents"],
        "supports_offload": False,
    },
    "qwen36_35b_uncensored_1m": {
        "name": "Qwen3.6 35B Uncensored (Q8, 1M ctx)",
        "description": "llama.cpp — Q8_K_P, 43GB, Qwen3.6 35B-A3B uncensored, YaRN-extended to 1M context, Q8_0 KV cache (~84GB VRAM total)",
        "port": 8098,
        "systemd_service": "qwen36-35b-uncensored-1m.service",
        "cmd": ["/mnt/raid1_nvme/models/ik_llama.cpp/build/bin/llama-server",
                "--model", "/mnt/raid1_nvme/models/qwen36-35b-uncensored/Qwen3.6-35B-A3B-Uncensored-HauhauCS-Aggressive-Q8_K_P.gguf",
                "--alias", "qwen36-35b-uncensored-1m",
                "--ctx-size", "1048576",
                "--rope-scaling", "yarn",
                "--yarn-orig-ctx", "262144",
                "-ctk", "q8_0", "-ctv", "q8_0",
                "-ngl", "99",
                "-b", "2048", "-ub", "2048",
                "--cache-ram", "0",
                "--host", "0.0.0.0",
                "--port", "8098",
                "--jinja",
                "--chat-template-kwargs", '{"enable_thinking":false}',
                "--threads", "16",
                "--chunk", "4096"],
        "cwd": "/mnt/raid1_nvme/models/ik_llama.cpp",
        "env": {},
        "protocol": "http",
        "category": "LLM",
        "icon": "zap",
        "color": "#f97316",
        "tags": ["text-to-text", "text-to-code", "agents"],
        "supports_offload": False,
    },
    "qwen36_40b_q6": {
        "name": "Qwen3.6-40B Claude 4.6 Opus (Q6_K)",
        "description": "llama.cpp — Q6_K, 25GB, DavidAU Claude-4.6-Opus Deckard Heretic NEo CODE Di IMatrix MAX, 256K ctx",
        "port": 9000,
        "systemd_service": "qwen36-40b-q6.service",
        "cmd": ["/mnt/raid1_nvme/models/ik_llama.cpp/build/bin/llama-server",
                "--model", "/mnt/raid1_nvme/models/davidau-qwen3.6-40b/Qwen3.6-40B-Deck-Opus-NEO-CODE-HERE-2T-OT-Q6_K.gguf",
                "--alias", "qwen36-40b-q6",
                "--ctx-size", "262144",
                "-ngl", "99",
                "-b", "2048", "-ub", "2048",
                "--host", "0.0.0.0",
                "--port", "8087",
                "--jinja",
                "--threads", "16"],
        "cwd": "/mnt/raid1_nvme/models/ik_llama.cpp",
        "env": {},
        "protocol": "http",
        "category": "LLM",
        "icon": "brain",
        "color": "#818cf8",
        "tags": ["text-to-text", "text-to-code", "agents", "claude"],
        "supports_offload": False,
    },
    "qwen36_40b_q8": {
        "name": "Qwen3.6-40B Claude 4.6 Opus (HIGH-Q8_0)",
        "description": "llama.cpp — HIGH-Q8_0, 38GB, DavidAU Claude-4.6-Opus Deckard Heretic NEo CODE Di IMatrix MAX, 256K ctx",
        "port": 9001,
        "systemd_service": "qwen36-40b-q8.service",
        "cmd": ["/mnt/raid1_nvme/models/ik_llama.cpp/build/bin/llama-server",
                "--model", "/mnt/raid1_nvme/models/davidau-qwen3.6-40b/Qwen3.6-40B-Deck-Opus-NEO-CODE-HERE-2T-OT-HIGH-Q8_0.gguf",
                "--alias", "qwen36-40b-q8",
                "--ctx-size", "262144",
                "-ngl", "99",
                "-b", "2048", "-ub", "2048",
                "--host", "0.0.0.0",
                "--port", "8088",
                "--jinja",
                "--threads", "16"],
        "cwd": "/mnt/raid1_nvme/models/ik_llama.cpp",
        "env": {},
        "protocol": "http",
        "category": "LLM",
        "icon": "brain-circuit",
        "color": "#a855f7",
        "tags": ["text-to-text", "text-to-code", "agents", "claude", "high-precision"],
        "supports_offload": False,
    },
    "qwen36_40b_q8_384k": {
        "name": "Qwen3.6-40B Claude 4.6 Opus (HIGH-Q8_0, 384K ctx)",
        "description": "llama.cpp — HIGH-Q8_0, 38GB, DavidAU Claude-4.6-Opus Deckard Heretic NEo CODE Di IMatrix MAX, YaRN-extended to 384K ctx, Q4_0 KV cache (~81GB VRAM total)",
        "port": 9002,
        "systemd_service": "qwen36-40b-q8-384k.service",
        "cmd": ["/mnt/raid1_nvme/models/ik_llama.cpp/build/bin/llama-server",
                "--model", "/mnt/raid1_nvme/models/davidau-qwen3.6-40b/Qwen3.6-40B-Deck-Opus-NEO-CODE-HERE-2T-OT-HIGH-Q8_0.gguf",
                "--alias", "qwen36-40b-q8-384k",
                "--ctx-size", "393216",
                "--rope-scaling", "yarn",
                "--yarn-orig-ctx", "262144",
                "-ctk", "q4_0", "-ctv", "q4_0",
                "-ngl", "99",
                "-b", "2048", "-ub", "2048",
                "--host", "0.0.0.0",
                "--port", "9002",
                "--jinja",
                "--threads", "16"],
        "cwd": "/mnt/raid1_nvme/models/ik_llama.cpp",
        "env": {},
        "protocol": "http",
        "category": "LLM",
        "icon": "brain-circuit",
        "color": "#a855f7",
        "tags": ["text-to-text", "text-to-code", "agents", "claude", "high-precision"],
        "supports_offload": False,
    },
    "qwen36_40b_q8_1m": {
        "name": "Qwen3.6-40B Claude 4.6 Opus (HIGH-Q8_0, 1M ctx)",
        "description": "llama.cpp — HIGH-Q8_0, 38GB, DavidAU Claude-4.6-Opus Deckard Heretic NEo CODE Di IMatrix MAX, YaRN-extended to 1M ctx, Q8_0 KV cache — needs the GPU to itself, will OOM if another large model is loaded",
        "port": 9004,
        "systemd_service": "qwen36-40b-q8-1m.service",
        "cmd": ["/mnt/raid1_nvme/models/ik_llama.cpp/build/bin/llama-server",
                "--model", "/mnt/raid1_nvme/models/davidau-qwen3.6-40b/Qwen3.6-40B-Deck-Opus-NEO-CODE-HERE-2T-OT-HIGH-Q8_0.gguf",
                "--alias", "qwen36-40b-q8-1m",
                "--ctx-size", "1048576",
                "--rope-scaling", "yarn",
                "--yarn-orig-ctx", "262144",
                "-ctk", "q8_0", "-ctv", "q8_0",
                "-ngl", "99",
                "-b", "2048", "-ub", "2048",
                "--host", "0.0.0.0",
                "--port", "9004",
                "--jinja",
                "--threads", "16"],
        "cwd": "/mnt/raid1_nvme/models/ik_llama.cpp",
        "env": {},
        "protocol": "http",
        "category": "LLM",
        "icon": "brain-circuit",
        "color": "#a855f7",
        "tags": ["text-to-text", "text-to-code", "agents", "claude", "high-precision"],
        "supports_offload": False,
    },
    "qwen36_40b_q8_512k": {
        "name": "Qwen3.6-40B Claude 4.6 Opus (HIGH-Q8_0, 512K ctx)",
        "description": "llama.cpp — HIGH-Q8_0, 38GB, DavidAU Claude-4.6-Opus Deckard Heretic NEo CODE Di IMatrix MAX, YaRN-extended to 512K ctx, Q8_0 KV cache — needs the GPU to itself, will OOM if another large model is loaded",
        "port": 9005,
        "systemd_service": "qwen36-40b-q8-512k.service",
        "cmd": ["/mnt/raid1_nvme/models/ik_llama.cpp/build/bin/llama-server",
                "--model", "/mnt/raid1_nvme/models/davidau-qwen3.6-40b/Qwen3.6-40B-Deck-Opus-NEO-CODE-HERE-2T-OT-HIGH-Q8_0.gguf",
                "--alias", "qwen36-40b-q8-512k",
                "--ctx-size", "524288",
                "--rope-scaling", "yarn",
                "--yarn-orig-ctx", "262144",
                "-ctk", "q8_0", "-ctv", "q8_0",
                "-ngl", "99",
                "-b", "2048", "-ub", "2048",
                "--host", "0.0.0.0",
                "--port", "9005",
                "--jinja",
                "--threads", "16"],
        "cwd": "/mnt/raid1_nvme/models/ik_llama.cpp",
        "env": {},
        "protocol": "http",
        "category": "LLM",
        "icon": "brain-circuit",
        "color": "#a855f7",
        "tags": ["text-to-text", "text-to-code", "agents", "claude", "high-precision"],
        "supports_offload": False,
    },
    "qwen36_40b_q6_512k": {
        "name": "Qwen3.6-40B Claude 4.6 Opus (Q6_K, 512K ctx)",
        "description": "llama.cpp — Q6_K, 25GB, DavidAU Claude-4.6-Opus Deckard Heretic NEo CODE Di IMatrix MAX, YaRN-extended to 512K ctx, Q4_0 KV cache (~83GB VRAM total)",
        "port": 9003,
        "systemd_service": "qwen36-40b-q6-512k.service",
        "cmd": ["/mnt/raid1_nvme/models/ik_llama.cpp/build/bin/llama-server",
                "--model", "/mnt/raid1_nvme/models/davidau-qwen3.6-40b/Qwen3.6-40B-Deck-Opus-NEO-CODE-HERE-2T-OT-Q6_K.gguf",
                "--alias", "qwen36-40b-q6-512k",
                "--ctx-size", "524288",
                "--rope-scaling", "yarn",
                "--yarn-orig-ctx", "262144",
                "-ctk", "q4_0", "-ctv", "q4_0",
                "-ngl", "99",
                "-b", "2048", "-ub", "2048",
                "--host", "0.0.0.0",
                "--port", "9003",
                "--jinja",
                "--threads", "16"],
        "cwd": "/mnt/raid1_nvme/models/ik_llama.cpp",
        "env": {},
        "protocol": "http",
        "category": "LLM",
        "icon": "brain",
        "color": "#818cf8",
        "tags": ["text-to-text", "text-to-code", "agents", "claude"],
        "supports_offload": False,
    },
    "qwen35_122b_q5": {
        "name": "Qwen3.5 122B Q5 (single)",
        "description": "llama.cpp — Q5_K_M, 88GB, max quality, single session",
        "port": 8082,
        "systemd_service": "qwen35-122b.service",
        "cmd": ["/mnt/raid1_nvme/models/ik_llama.cpp/build/bin/llama-server",
                "--model", "/models/qwen35-122b/Qwen_Qwen3.5-122B-A10B-Q5_K_M/Qwen_Qwen3.5-122B-A10B-Q5_K_M-00001-of-00003.gguf",
                "--alias", "qwen35-122b",
                "--ctx-size", "262144",
                "-ngl", "99",
                "-b", "2048", "-ub", "2048",
                "--host", "0.0.0.0",
                "--port", "8082",
                "--jinja",
                "--threads", "16"],
        "cwd": "/mnt/raid1_nvme/models/ik_llama.cpp",
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
        "cmd": ["/mnt/raid1_nvme/models/ik_llama.cpp/build/bin/llama-server",
                "--model", "/mnt/raid1_nvme/models/qwen35-122b/Qwen_Qwen3.5-122B-A10B-Q4_K_M/Qwen_Qwen3.5-122B-A10B-Q4_K_M-00001-of-00002.gguf",
                "--alias", "qwen35-122b",
                "--ctx-size", "262144",
                "--parallel", "4",
                "-ngl", "99",
                "-b", "2048", "-ub", "2048",
                "--host", "0.0.0.0",
                "--port", "8083",
                "--jinja",
                "--threads", "16"],
        "cwd": "/mnt/raid1_nvme/models/ik_llama.cpp",
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
        "cmd": ["/mnt/raid1_nvme/models/ik_llama.cpp/build/bin/llama-server",
                "--model", "/mnt/raid1_nvme/models/minimax-m2.5/smol-IQ3_KS/MiniMax-M2.5-smol-IQ3_KS-00001-of-00003.gguf",
                "--alias", "minimax-m2.5",
                "--ctx-size", "65536",
                "-ngl", "99",
                "-b", "2048", "-ub", "2048",
                "--host", "0.0.0.0",
                "--port", "8081",
                "--jinja",
                "--threads", "16"],
        "cwd": "/mnt/raid1_nvme/models/ik_llama.cpp",
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
    "unlimited_ocr": {
        "name": "Unlimited-OCR",
        "description": "One-shot long-horizon document parsing (Baidu/SGLang)",
        "port": 10000,
        "cmd": ["/mnt/raid1_nvme/JanusPro7b/Unlimited-OCR/unlimited-ocr-venv/bin/python", "infer.py"],
        "cwd": "/mnt/raid1_nvme/JanusPro7b/Unlimited-OCR",
        "env": {},
        "protocol": "http",
        "category": "Tools",
        "icon": "scan-line",
        "color": "#06b6d4",
        "tags": ["ocr", "document-parsing", "image-to-text"],
        "supports_offload": False,
        "vram_gb": 24,
    },
    "gemma_4_31b": {
        "name": "Gemma 4 31B IT",
        "description": "Google Gemma 4 31B Instruct - Advanced reasoning & generation",
        "port": 8091,
        "cmd": ["/mnt/raid1_nvme/models/gemma-4-31b-it/venv/bin/python", "-m", "transformers"],
        "cwd": "/mnt/raid1_nvme/models/gemma-4-31b-it",
        "env": {"CUDA_VISIBLE_DEVICES": "0"},
        "protocol": "http",
        "category": "LLM",
        "icon": "brain",
        "color": "#8b5cf6",
        "tags": ["llm", "instruct", "reasoning"],
        "supports_offload": True,
        "vram_gb": 66,
    },
    "qwen36_27b_nvfp4": {
        "name": "Qwen3.6 27B NVFP4",
        "description": "Unsloth Qwen3.6 27B NVFP4 quantized - Fast vLLM inference",
        "port": 8088,
        "systemd_service": "qwen36-27b-nvfp4.service",
        "cmd": ["/mnt/raid1_nvme/models/Qwen3.6-27B-NVFP4/venv/bin/vllm", "serve", "unsloth/Qwen3.6-27B-NVFP4",
                "--port", "8088", "--host", "0.0.0.0",
                "--gpu-memory-utilization", "0.90",
                "--max-num-seqs", "256"],
        "cwd": "/mnt/raid1_nvme/models/Qwen3.6-27B-NVFP4",
        "env": {"CUDA_VISIBLE_DEVICES": "0", "FLASHINFER_DISABLE_VERSION_CHECK": "1"},
        "protocol": "http",
        "category": "LLM",
        "icon": "cpu",
        "color": "#06b6d4",
        "tags": ["llm", "quantized", "fast"],
        "supports_offload": True,
        "vram_gb": 20,
    },
    "qwen38_27b_nvfp4": {
        "name": "Qwen3.8 27B NVFP4",
        "description": "Unsloth Qwen3.8 27B NVFP4 quantized - ~1.5x faster than BF16, vLLM, tool calling, 256K ctx",
        "port": 8094,
        "systemd_service": "qwen38-27b-nvfp4.service",
        "cmd": ["/mnt/raid1_nvme/models/Qwen3.8-27B-NVFP4/venv/bin/vllm", "serve", "unsloth/Qwen3.8-27B-NVFP4",
                "--port", "8094", "--host", "0.0.0.0",
                "--gpu-memory-utilization", "0.45",
                "--max-model-len", "262144",
                "--max-num-seqs", "64",
                "--enable-auto-tool-choice", "--tool-call-parser", "hermes"],
        "cwd": "/mnt/raid1_nvme/models/Qwen3.8-27B-NVFP4",
        "env": {"CUDA_VISIBLE_DEVICES": "0", "FLASHINFER_DISABLE_VERSION_CHECK": "1"},
        "protocol": "http",
        "category": "LLM",
        "icon": "cpu",
        "color": "#0ea5e9",
        "tags": ["llm", "quantized", "fast"],
        "supports_offload": True,
        "vram_gb": 20,
    },
    "ornith_35b_q8": {
        "name": "Ornith 1.0 35B Q8_0",
        "description": "Ornith 1.0 35B 8-bit GGUF - High quality quantized",
        "port": 8092,
        "cmd": ["/mnt/raid1_nvme/models/Ornith-1.0-35B/venv/bin/python", "-m", "llama_cpp.server", "--model", "/mnt/raid1_nvme/models/Ornith-1.0-35B/models/ornith-1.0-35b-Q8_0.gguf", "--port", "8092"],
        "cwd": "/mnt/raid1_nvme/models/Ornith-1.0-35B",
        "env": {"CUDA_VISIBLE_DEVICES": "0"},
        "protocol": "http",
        "category": "LLM",
        "icon": "brain",
        "color": "#f59e0b",
        "tags": ["llm", "gguf", "quantized"],
        "supports_offload": True,
        "vram_gb": 38,
    },
    "ornith_35b_bf16": {
        "name": "Ornith 1.0 35B BF16",
        "description": "Ornith 1.0 35B 16-bit bfloat16 - Full precision",
        "port": 8090,
        "cmd": ["/mnt/raid1_nvme/models/Ornith-1.0-35B/venv/bin/python", "-m", "llama_cpp.server", "--model", "/mnt/raid1_nvme/models/Ornith-1.0-35B/models/ornith-1.0-35b-bf16.gguf", "--port", "8090"],
        "cwd": "/mnt/raid1_nvme/models/Ornith-1.0-35B",
        "env": {"CUDA_VISIBLE_DEVICES": "0"},
        "protocol": "http",
        "category": "LLM",
        "icon": "cpu",
        "color": "#ef4444",
        "tags": ["llm", "gguf", "full-precision"],
        "supports_offload": True,
        "vram_gb": 72,
    },
    "laguna_s21_q4": {
        "name": "Laguna S-2.1 Q4_K_M",
        "description": "Poolside Laguna S-2.1 Q4 - 1M context, 32B parameter model",
        "port": 8095,
        "cmd": ["/mnt/raid1_nvme/models/poolside-s21-venv/bin/python", "-m", "llama_cpp.server", "--model", "/mnt/raid1_nvme/models/Laguna-S-2.1/models/laguna-s-2.1-Q4_K_M.gguf", "--port", "8095"],
        "cwd": "/mnt/raid1_nvme/models/Laguna-S-2.1",
        "env": {"CUDA_VISIBLE_DEVICES": "0"},
        "protocol": "http",
        "category": "LLM",
        "icon": "brain",
        "color": "#7c3aed",
        "tags": ["llm", "gguf", "1m-context"],
        "supports_offload": True,
        "vram_gb": 96,
    },
    "laguna_s21_q8": {
        "name": "Laguna S-2.1 Q8_0",
        "description": "Poolside Laguna S-2.1 Q8 - 1M context, 32B parameter model (high quality)",
        "port": 8096,
        "cmd": ["/mnt/raid1_nvme/models/poolside-s21-venv/bin/python", "-m", "llama_cpp.server", "--model", "/mnt/raid1_nvme/models/Laguna-S-2.1/models/laguna-s-2.1-Q8_0.gguf", "--port", "8096"],
        "cwd": "/mnt/raid1_nvme/models/Laguna-S-2.1",
        "env": {"CUDA_VISIBLE_DEVICES": "0"},
        "protocol": "http",
        "category": "LLM",
        "icon": "brain-circuit",
        "color": "#4f46e5",
        "tags": ["llm", "gguf", "1m-context", "high-quality"],
        "supports_offload": True,
        "vram_gb": 96,
    },
    "minimax_h3": {
        "name": "MiniMax H3",
        "description": "33B omni-modal image-text-to-video generation with native stereo audio (BF16, vLLM)",
        "port": 30010,
        "systemd_service": "minimax-h3.service",
        "cmd": ["/usr/local/bin/vllm", "serve", "MiniMaxAI/MiniMax-H3", "--port", "30010", "--model-variant", "fl2va", "--dtype", "bfloat16", "--cpu-offload-gb", "16"],
        "cwd": "/mnt/raid1_nvme/models/MiniMax-H3",
        "env": {},
        "protocol": "http",
        "category": "Video",
        "icon": "film",
        "color": "#f97316",
        "tags": ["image-text-to-video", "text-to-video", "audio-video", "33b"],
        "supports_offload": True,
        "vram_gb": 96,
    },
 
    "qwen36_27b_fable": {
        "name": "Qwen3.6-27B Fable Fusion 711 (Q8_0 MTP)",
        "description": "llama.cpp — Q8_0 MTP, 30GB, DavidAU Heretic Uncensored, NEO MAX Imatrix, vision-enabled",
        "port": 9006,
        "systemd_service": "qwen36-27b-fable.service",
        "cmd": ["/mnt/raid1_nvme/models/ik_llama.cpp/build/bin/llama-server",
                "--model", "/mnt/raid1_nvme/models/qwen36-27b-fable-fusion/Qwen3.6-27B-Fable-Fus-711-UnHeretic-NM-DAU-NEO-MAX-NEO-MTP-Q8_0.gguf",
                "--alias", "qwen36-27b-fable",
                "--ctx-size", "262144",
                "-ngl", "99",
                "-b", "2048", "-ub", "2048",
                "--host", "0.0.0.0",
                "--port", "9006",
                "--jinja",
                "--threads", "16"],
        "cwd": "/mnt/raid1_nvme/models/ik_llama.cpp",
        "env": {},
        "protocol": "http",
        "category": "LLM",
        "icon": "brain-circuit",
        "color": "#a855f7",
        "tags": ["text-to-text", "vision", "q8_0", "mtp", "heretic"],
        "supports_offload": False,
        "vram_gb": 32,
    },
    "qwen36_27b_fable_1m": {
        "name": "Qwen3.6-27B Fable Fusion 711 (Q8_0 MTP, 1M ctx)",
        "description": "llama.cpp — Q8_0 MTP, 30GB, DavidAU Heretic Uncensored, YaRN-extended to 1M ctx, Q8_0 KV cache",
        "port": 9007,
        "systemd_service": "qwen36-27b-fable-1m.service",
        "cmd": ["/mnt/raid1_nvme/models/ik_llama.cpp/build/bin/llama-server",
                "--model", "/mnt/raid1_nvme/models/qwen36-27b-fable-fusion/Qwen3.6-27B-Fable-Fus-711-UnHeretic-NM-DAU-NEO-MAX-NEO-MTP-Q8_0.gguf",
                "--alias", "qwen36-27b-fable-1m",
                "--ctx-size", "1048576",
                "--rope-scaling", "yarn",
                "--yarn-orig-ctx", "262144",
                "-ctk", "q8_0", "-ctv", "q8_0",
                "-ngl", "99",
                "-b", "2048", "-ub", "2048",
                "--host", "0.0.0.0",
                "--port", "9007",
                "--jinja",
                "--threads", "16"],
        "cwd": "/mnt/raid1_nvme/models/ik_llama.cpp",
        "env": {},
        "protocol": "http",
        "category": "LLM",
        "icon": "brain-circuit",
        "color": "#a855f7",
        "tags": ["text-to-text", "vision", "q8_0", "mtp", "heretic", "1m-context"],
        "supports_offload": False,
        "vram_gb": 96,
      },
    "qwen36_27b_fable_q6": {
        "name": "Qwen3.6-27B Fable Fusion 711 (Q6_K MTP)",
        "description": "llama.cpp — Q6_K MTP, ~20GB, DavidAU Heretic Uncensored, NEO MAX Imatrix, vision-enabled",
        "port": 9009,
        "systemd_service": "qwen36-27b-fable-q6.service",
        "cmd": ["/mnt/raid1_nvme/models/ik_llama.cpp/build/bin/llama-server",
                "--model", "/mnt/raid1_nvme/models/qwen36-27b-fable-fusion/Qwen3.6-27B-Fable-Fus-711-UnHeretic-NM-DAU-NEO-MAX-NEO-MTP-Q6_K.gguf",
                "--alias", "qwen36-27b-fable-q6",
                "--ctx-size", "262144",
                "-ngl", "99",
                "-b", "2048", "-ub", "2048",
                "--host", "0.0.0.0",
                "--port", "9009",
                "--jinja",
                "--threads", "16"],
        "cwd": "/mnt/raid1_nvme/models/ik_llama.cpp",
        "env": {},
        "protocol": "http",
        "category": "LLM",
        "icon": "brain-circuit",
        "color": "#a855f7",
        "tags": ["text-to-text", "vision", "q6_k", "mtp", "heretic"],
        "supports_offload": False,
        "vram_gb": 24,
    },
    "qwen36_27b_fable_amd": {
        "name": "Qwen3.6-27B Fable Fusion 711 (AMD IQ4_XS MTP)",
        "description": "llama.cpp — AMD-optimized IQ4_XS MTP, 16GB, DavidAU Heretic Uncensored, NEO MAX Imatrix, vision-enabled",
        "port": 9008,
        "systemd_service": "qwen36-27b-fable-amd.service",
        "cmd": ["/mnt/raid1_nvme/models/ik_llama.cpp/build/bin/llama-server",
                "--model", "/mnt/raid1_nvme/models/qwen36-27b-fable-amd/Qwen3.6-27B-Fable-Fus-711-UnHeretic-NM-DAU-NEO-MAX-NEO-AMD-MTP-IQ4_XS.gguf",
                "--alias", "qwen36-27b-fable-amd",
                "--ctx-size", "262144",
                "-ngl", "99",
                "-b", "2048", "-ub", "2048",
                "--host", "0.0.0.0",
                "--port", "9008",
                "--jinja",
                "--threads", "16"],
        "cwd": "/mnt/raid1_nvme/models/ik_llama.cpp",
        "env": {},
        "protocol": "http",
        "category": "LLM",
        "icon": "cpu",
        "color": "#f97316",
        "tags": ["text-to-text", "vision", "iq4_xs", "mtp", "heretic", "amd-optimized"],
        "supports_offload": False,
        "vram_gb": 18,
    },
    "vllm_qwen38_27b": {
        "name": "Qwen3.8 27B",
        "description": "vLLM OpenAI-compatible API (Qwen3.8 27B, Qwen3.5 hybrid-attn arch, 256K context — coding & chat)",
        "port": 8010,
        "systemd_service": "qwen38-27b.service",
        "cmd": ["/mnt/raid1_nvme/Qwen3_8-27B/venv/bin/python", "-m", "vllm.entrypoints.openai.api_server",
                "--model", "/mnt/raid1_nvme/Qwen3_8-27B/fp16",
                "--served-model-name", "qwen38-27b", "qwen38-27b-fp16",
                "--enable-auto-tool-choice", "--tool-call-parser", "qwen3_coder",
                "--chat-template", "/mnt/raid1_nvme/Qwen3_8-27B/templates/sharp-chat-template-v22.1.1.jinja",
                "--dtype", "half",
                "--port", "8010", "--host", "0.0.0.0",
                "--gpu-memory-utilization", "0.90",
                "--max-model-len", "262144",
                "--max-num-seqs", "256",
                "--speculative-config", '{"method": "mtp", "num_speculative_tokens": 2}',
                "--default-chat-template-kwargs", '{"enable_thinking": true, "reasoning_effort": "medium"}'],
        "cwd": "/mnt/raid1_nvme/Qwen3_8-27B",
        "env": {"PYTORCH_ALLOC_CONF": "expandable_segments:True"},
        "protocol": "http",
        "category": "LLM",
        "icon": "sparkles",
        "color": "#f97316",
        "tags": ["text-to-text", "text-to-code"],
        "supports_offload": True,
        "vram_gb": 80,
    },
    "qwen38_flash_next_iq4xs": {
        "name": "Qwen3.8-Flash-Next 125B (IQ4_XS)",
        "description": "llama.cpp PR 27742 (Qwen4Exp) — 125B MoE multimodal, ~94GB IQ4_XS, PLE n-gram hash table, 262K ctx, GPU+RAM hybrid",
        "port": 9010,
        "systemd_service": "qwen38-flash-next.service",
        "cmd": ["/mnt/raid1_nvme/models/llama.cpp-pr27742/build/bin/llama-server",
                "--model", "/mnt/raid1_nvme/models/qwen38-flash-next/UD-IQ4_XS/Qwen3.8-Flash-Next-UD-IQ4_XS.gguf",
                "--alias", "qwen38-flash-next",
                "--ctx-size", "262144",
                "-ngl", "99",
                "-b", "2048", "-ub", "2048",
                "--host", "0.0.0.0",
                "--port", "9010",
                "--jinja",
                "--temp", "1.0", "--top-p", "0.95", "--top-k", "20",
                "--threads", "16"],
        "cwd": "/mnt/raid1_nvme/models/llama.cpp-pr27742",
        "env": {"GGML_CUDA_NO_PINNED": "1",
                "LD_LIBRARY_PATH": "/mnt/raid1_nvme/models/llama.cpp-pr27742/build/src:/mnt/raid1_nvme/models/llama.cpp-pr27742/build/ggml/src:/mnt/raid1_nvme/models/llama.cpp-pr27742/build/examples/mtmd"},
        "protocol": "http",
        "category": "LLM",
        "icon": "brain-circuit",
        "color": "#06b6d4",
        "tags": ["text-to-text", "vision", "iq4_xs", "moe", "125b", "qwen4exp", "262k-context"],
        "supports_offload": False,
        "vram_gb": 94,
    },
    "qwen38_flash_next_q4kxl": {
        "name": "Qwen3.8-Flash-Next 125B (Q4_K_XL)",
        "description": "llama.cpp PR 27742 (Qwen4Exp) — 125B MoE multimodal, ~111GB Q4_K_XL (Unsloth recommended), PLE n-gram hash table, 262K ctx, GPU+RAM hybrid",
        "port": 9011,
        "systemd_service": "qwen38-flash-next-q4kxl.service",
        "cmd": ["/mnt/raid1_nvme/models/llama.cpp-pr27742/build/bin/llama-server",
                "--model", "/mnt/raid1_nvme/models/qwen38-flash-next/UD-Q4_K_XL/Qwen3.8-Flash-Next-UD-Q4_K_XL.gguf",
                "--alias", "qwen38-flash-next-q4kxl",
                "--ctx-size", "262144",
                "-ngl", "99",
                "-b", "2048", "-ub", "2048",
                "--host", "0.0.0.0",
                "--port", "9011",
                "--jinja",
                "--temp", "1.0", "--top-p", "0.95", "--top-k", "20",
                "--threads", "16"],
        "cwd": "/mnt/raid1_nvme/models/llama.cpp-pr27742",
        "env": {"GGML_CUDA_NO_PINNED": "1",
                "LD_LIBRARY_PATH": "/mnt/raid1_nvme/models/llama.cpp-pr27742/build/src:/mnt/raid1_nvme/models/llama.cpp-pr27742/build/ggml/src:/mnt/raid1_nvme/models/llama.cpp-pr27742/build/examples/mtmd"},
        "protocol": "http",
        "category": "LLM",
        "icon": "brain-circuit",
        "color": "#0ea5e9",
        "tags": ["text-to-text", "vision", "q4_k_xl", "moe", "125b", "qwen4exp", "262k-context"],
        "supports_offload": False,
        "vram_gb": 111,
    },
    "vllm_qwen38_27b_q8": {
        "name": "Qwen3.8 27B (Q8_K_XL)",
        "description": "llama.cpp — unsloth Q8_K_XL GGUF, ~30GB, 256K ctx — coding & chat",
        "port": 8011,
        "systemd_service": "qwen38-27b-q8.service",
        "cmd": ["/mnt/raid1_nvme/Qwen3_8-27B/llama.cpp/build/bin/llama-server",
                "--model", "/mnt/raid1_nvme/Qwen3_8-27B/gguf-q8/Qwen3.8-27B-UD-Q8_K_XL.gguf",
                "--spec-draft-model", "/mnt/raid1_nvme/Qwen3_8-27B/gguf-q8/mtp/MTP/mtp-Qwen3.8-27B-Q4_0.gguf",
                "--spec-type", "draft-mtp",
                "--spec-draft-ngl", "99",
                "--chat-template-file", "/mnt/raid1_nvme/Qwen3_8-27B/templates/sharp-chat-template-v22.1.1.jinja",
                "--alias", "qwen38-27b-q8",
                "--host", "0.0.0.0",
                "--port", "8011",
                "--ctx-size", "262144",
                "--parallel", "1",
                "-ngl", "99",
                "--reasoning-effort", "medium",
                "--threads", "16"],
        "cwd": "/mnt/raid1_nvme/Qwen3_8-27B",
        "env": {},
        "protocol": "http",
        "category": "LLM",
        "icon": "sparkles",
        "color": "#f97316",
        "tags": ["text-to-text", "text-to-code"],
        "supports_offload": False,
        "vram_gb": 46,
    },
    "vllm_qwen38_27b_q6": {
        "name": "Qwen3.8 27B (Q6_K_XL)",
        "description": "llama.cpp — unsloth Q6_K_XL GGUF (Dynamic 3.0), ~24GB, 256K ctx, MTP speculative decoding — fastest variant (~105 tok/s)",
        "port": 8016,
        "systemd_service": "qwen38-27b-q6.service",
        "cmd": ["/mnt/raid1_nvme/Qwen3_8-27B/llama.cpp/build/bin/llama-server",
                "--model", "/mnt/raid1_nvme/Qwen3_8-27B/gguf-q6/Qwen3.8-27B-UD-Q6_K_XL.gguf",
                "--spec-draft-model", "/mnt/raid1_nvme/Qwen3_8-27B/gguf-q8/mtp/MTP/mtp-Qwen3.8-27B-Q4_0.gguf",
                "--spec-type", "draft-mtp",
                "--spec-draft-ngl", "99",
                "--chat-template-file", "/mnt/raid1_nvme/Qwen3_8-27B/templates/sharp-chat-template-v22.1.1.jinja",
                "--alias", "qwen38-27b-q6",
                "--host", "0.0.0.0",
                "--port", "8016",
                "--ctx-size", "262144",
                "--parallel", "1",
                "-ngl", "99",
                "--reasoning-effort", "medium",
                "--threads", "16"],
        "cwd": "/mnt/raid1_nvme/Qwen3_8-27B",
        "env": {},
        "protocol": "http",
        "category": "LLM",
        "icon": "sparkles",
        "color": "#f97316",
        "tags": ["text-to-text", "text-to-code"],
        "supports_offload": False,
        "vram_gb": 40,
    },
    "vllm_qwen38_27b_uncensored": {
        "name": "Qwen3.8 27B Uncensored",
        "description": "vLLM OpenAI-compatible API (orcarouter abliterated finetune, FP16, MTP speculative decoding, 256K ctx)",
        "port": 8017,
        "systemd_service": "qwen38-27b-uncensored.service",
        "cmd": ["/mnt/raid1_nvme/Qwen3_8-27B/venv/bin/python", "-m", "vllm.entrypoints.openai.api_server",
                "--model", "/mnt/raid1_nvme/Qwen3_8-27B/uncensored",
                "--served-model-name", "qwen38-27b-uncensored",
                "--enable-auto-tool-choice", "--tool-call-parser", "qwen3_coder",
                "--chat-template", "/mnt/raid1_nvme/Qwen3_8-27B/templates/sharp-chat-template-v22.1.1.jinja",
                "--dtype", "half",
                "--port", "8017", "--host", "0.0.0.0",
                "--gpu-memory-utilization", "0.90",
                "--max-model-len", "262144",
                "--max-num-seqs", "256",
                "--speculative-config", '{"method": "mtp", "num_speculative_tokens": 2}',
                "--default-chat-template-kwargs", '{"enable_thinking": true, "reasoning_effort": "medium"}'],
        "cwd": "/mnt/raid1_nvme/Qwen3_8-27B",
        "env": {"PYTORCH_ALLOC_CONF": "expandable_segments:True"},
        "protocol": "http",
        "category": "LLM",
        "icon": "sparkles",
        "color": "#f97316",
        "tags": ["text-to-text", "text-to-code", "uncensored"],
        "supports_offload": True,
        "vram_gb": 56,
    },
}

# Group ordering for display
CATEGORY_ORDER = ["Image", "Video", "Audio", "LLM", "Tools"]

processes: dict[str, subprocess.Popen] = {}
log_files: dict[str, object] = {}
op_lock = threading.Lock()

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

app = FastAPI()

# ── Power tracking ────────────────────────────────────────────────
import threading, json as _json_mod, datetime as _dt_mod

POWER_LOG = "/mnt/raid1_nvme/JanusPro7b/logs/power_usage.json"

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

def _power_sampler():
    """Sample GPU power every 60s and accumulate watt-hours per day."""
    import time
    while True:
        try:
            out = subprocess.run(
                ["nvidia-smi", "--query-gpu=power.draw", "--format=csv,noheader,nounits"],
                capture_output=True, text=True, timeout=5,
            )
            watts = float(out.stdout.strip())
            today = _dt_mod.date.today().isoformat()
            data = _load_power_log()
            # watt-hours = watts * (1min / 60min)
            data[today] = data.get(today, 0) + watts / 60.0
            _save_power_log(data)
        except Exception:
            pass
        time.sleep(60)

_power_thread = threading.Thread(target=_power_sampler, daemon=True)
_power_thread.start()


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

    processes.pop(model_id, None)
    systemd_service = model.get("systemd_service")
    if systemd_service:
        result = subprocess.run(
            ["systemctl", "is-active", systemd_service], capture_output=True, text=True
        )
        active_state = result.stdout.strip()
        if active_state in ("active", "activating", "reloading"):
            return {"status": "ready" if port_up else "starting", "pid": None, "managed": False}
        return {"status": "stopped", "pid": None, "managed": False}

    if port_up:
        return {"status": "ready", "pid": None, "managed": False}
    return {"status": "stopped", "pid": None, "managed": False}


@app.get("/api/power")
def api_power():
    """Return historical watt-hour usage per day."""
    data = _load_power_log()
    sorted_data = dict(sorted(data.items()))
    today = _dt_mod.date.today().isoformat()
    today_wh = data.get(today, 0)
    return JSONResponse({"by_day": sorted_data, "today_wh": today_wh})


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


QUANT_PATTERNS = [
    ("NVFP4", "NVFP4"),
    ("IQ3_KS", "IQ3_KS"),
    ("IQ4_XS", "IQ4_XS"),
    ("Q8_K_P", "Q8_K_P"),
    ("Q8_K_XL", "Q8_K_XL"),
    ("Q8_0", "Q8_0"),
    ("Q6_K", "Q6_K"),
    ("Q5_K_M", "Q5_K_M"),
    ("Q4_K_M", "Q4_K_M"),
    ("Q4_0", "Q4_0"),
    ("BF16", "BF16"),
    ("FP16", "FP16"),
    ("FP8", "FP8"),
]


def get_quant(model: dict) -> str:
    """Best-effort detection of the model's quantization/precision."""
    text = " ".join(
        [
            model.get("name", ""),
            model.get("description", ""),
            " ".join(model.get("cmd", [])),
        ]
    ).upper()
    for pattern, label in QUANT_PATTERNS:
        if pattern in text:
            return label
    cmd_args = " ".join(model.get("cmd", []))
    if "--dtype" in cmd_args:
        if "half" in cmd_args or "float16" in cmd_args or "fp16" in cmd_args:
            return "FP16"
        if "bfloat16" in cmd_args or "bf16" in cmd_args:
            return "BF16"
    if re.search(r"(?m)^--bf16\b", "\n".join(model.get("cmd", []))):
        return "BF16"
    return ""


@app.get("/api/status")
def api_status():
    try:
        token_stats = _load_token_stats()
        by_model = token_stats.get("by_model", {})
    except Exception:
        by_model = {}
    result = {}
    for model_id, model in MODELS.items():
        usage = by_model.get(model_id, {}).get("total", 0)
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
            "token_usage": usage,
            "quant": get_quant(model),
            "vram_gb": model.get("vram_gb"),
            **get_model_status(model_id),
        }
    return JSONResponse(result)


@app.post("/api/start/{model_id}")
def api_start(model_id: str, offload: bool = False):
    if model_id not in MODELS:
        return JSONResponse({"error": "Unknown model"}, status_code=404)

    with op_lock:
        status = get_model_status(model_id)
        if status["status"] != "stopped":
            return JSONResponse({"message": "Already running", **status})

        model = MODELS[model_id]

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

        log_path = f"/mnt/raid1_nvme/JanusPro7b/logs/{model_id}.log"
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
                os.kill(pid, signal.SIGKILL)
            except (ProcessLookupError, PermissionError):
                pass
    except Exception:
        pass


@app.post("/api/stop/{model_id}")
def api_stop(model_id: str):
    if model_id not in MODELS:
        return JSONResponse({"error": "Unknown model"}, status_code=404)

    with op_lock:
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
    log_path = f"/mnt/raid1_nvme/JanusPro7b/logs/{model_id}.log"
    text = ""
    try:
        with open(log_path) as f:
            lines = f.readlines()
            text = "".join(lines[-200:])
    except FileNotFoundError:
        pass
    if not text:
        svc = MODELS[model_id].get("systemd_service")
        if svc:
            try:
                out = subprocess.run(
                    ["journalctl", "-u", svc, "-n", "200", "--no-pager"],
                    capture_output=True, text=True, timeout=10,
                )
                text = out.stdout
            except Exception:
                pass
    return JSONResponse({"logs": text})


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
        # build_custom_entry returns (entry_id, entry) — id already computed above for the dup check
        _, entry = build_custom_entry(fid, variant, resolved, body, port)
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

  .table-bar {
    max-width: 1200px;
    margin: 24px auto 12px;
    padding: 0 20px;
    display: flex;
    align-items: center;
    gap: 12px;
  }
  .search-input {
    flex: 1;
    max-width: 420px;
    background: #1a1b23;
    border: 1px solid #27272a;
    border-radius: 8px;
    color: #e4e4e7;
    padding: 9px 14px;
    font-size: 0.9em;
    font-family: inherit;
    outline: none;
  }
  .search-input:focus { border-color: #667eea; }
  .search-input::placeholder { color: #52525b; }
  .table-count {
    margin-left: auto;
    color: #52525b;
    font-size: 0.8em;
    font-weight: 600;
    white-space: nowrap;
  }
  .table-wrap {
    max-width: 1200px;
    margin: 0 auto;
    padding: 0 20px 12px;
  }
  .table-card {
    background: #1a1b23;
    border: 1px solid #27272a;
    border-radius: 12px;
    overflow-x: auto;
  }
  #model-table {
    width: 100%;
    table-layout: fixed;
    border-collapse: collapse;
    font-size: 0.85em;
  }
  #model-table thead th:nth-child(1) { width: 170px; }
  #model-table thead th:nth-child(2) { width: 74px; }
  #model-table thead th:nth-child(3) { width: 80px; }
  #model-table thead th:nth-child(4) { width: 60px; }
  #model-table thead th:nth-child(5) { width: 130px; }
  #model-table thead th:nth-child(7) { width: 84px; }
  #model-table thead th:nth-child(8) { width: 76px; }
  #model-table thead th:nth-child(9) { width: 84px; }
  #model-table thead th:nth-child(10) { width: 56px; }
  #model-table thead th:nth-child(11) { width: 100px; }
  #model-table thead th {
    text-align: left;
    padding: 12px 10px;
    font-size: 0.7em;
    font-weight: 700;
    text-transform: uppercase;
    letter-spacing: 0.07em;
    color: #71717a;
    border-bottom: 1px solid #27272a;
    cursor: pointer;
    user-select: none;
    white-space: nowrap;
  }
  #model-table thead th:hover { color: #e4e4e7; }
  #model-table thead th.sorted-asc::after { content: ' \25b2'; color: #667eea; }
  #model-table thead th.sorted-desc::after { content: ' \25bc'; color: #667eea; }
  #model-table tbody td {
    padding: 9px 10px;
    border-bottom: 1px solid #202027;
    vertical-align: middle;
  }
  #model-table tbody tr.model-row:hover td { background: #1f2028; }
  .td-name { font-weight: 600; overflow-wrap: break-word; }
  .row-icon {
    display: inline-flex;
    width: 22px; height: 22px;
    border-radius: 6px;
    align-items: center; justify-content: center;
    font-size: 11px; font-weight: 700;
    margin-right: 8px;
  }
  .name-link { color: #e4e4e7; text-decoration: none; }
  .name-link.active:hover { color: #818cf8; text-decoration: underline; }
  .mono { font-family: 'JetBrains Mono', 'Fira Code', monospace; color: #a1a1aa; font-size: 0.92em; }
  .td-tags { white-space: normal; line-height: 1.7; }
  .td-tags .tag-badge { margin: 1px 3px 1px 0; }
  .cat-badge {
    display: inline-block;
    padding: 2px 8px;
    border-radius: 4px;
    font-size: 0.78em;
    font-weight: 700;
    white-space: nowrap;
    margin-right: 8px;
  }
  .td-desc { overflow-wrap: break-word; }
  .desc-text { color: #71717a; font-size: 0.88em; line-height: 1.35; }
  .dim { color: #3f3f46; }
  .td-status { white-space: normal; }
  .tag-badge {
    display: inline-block;
    background: #1e1e2e; color: #71717a;
    padding: 2px 8px; border-radius: 4px;
    font-size: 0.7em; font-weight: 500;
    border: 1px solid #27272a;
  }
  .status-dot { width: 9px; height: 9px; border-radius: 50%; flex-shrink: 0; margin-right: 6px; vertical-align: middle; }
  .status-dot.stopped { background: #52525b; }
  .status-dot.starting { background: #facc15; animation: pulse 1.5s infinite; }
  .status-dot.ready { background: #22c55e; box-shadow: 0 0 8px #22c55e55; }
  @keyframes pulse { 0%, 100% { opacity: 1; } 50% { opacity: 0.3; } }
  .status-text {
    font-size: 0.78em; color: #a1a1aa; font-weight: 600;
    text-transform: uppercase; letter-spacing: 0.06em;
  }
  .switch {
    position: relative; width: 36px; height: 20px; appearance: none;
    background: #27272a; border-radius: 10px; cursor: pointer; transition: background 0.2s;
    outline: none; border: 1px solid #3f3f46; flex-shrink: 0; margin: 0; vertical-align: middle;
  }
  .switch:checked { background: #22c55e; border-color: #22c55e; }
  .switch::after {
    content: ''; position: absolute; top: 2px; left: 2px;
    width: 14px; height: 14px; border-radius: 50%;
    background: #e4e4e7; transition: transform 0.2s;
  }
  .switch:checked::after { transform: translateX(16px); }
  .switch:disabled { opacity: 0.35; cursor: not-allowed; }
  .star {
    background: none; border: none; padding: 2px 6px;
    font-size: 1.15em; line-height: 1; cursor: pointer;
    color: #3f3f46; transition: color 0.15s, transform 0.1s;
  }
  .star:hover { color: #facc15; transform: scale(1.15); }
  .star.on { color: #facc15; text-shadow: 0 0 10px #facc1566; }
  .log-row td { padding: 0 10px 12px; background: #15161c; border-bottom: 1px solid #202027; }
  .row-logbox {
    background: #0f1117; border: 1px solid #27272a;
    border-radius: 8px; padding: 10px;
    font-family: 'JetBrains Mono', 'Fira Code', monospace;
    font-size: 0.72em; color: #9ca3af;
    max-height: 240px; overflow-y: auto;
    white-space: pre-wrap; word-break: break-all;
  }
  .alllogs {
    max-width: 1200px;
    margin: 4px auto 24px;
    padding: 0 20px;
  }
  .alllogs-panel {
    background: #1a1b23;
    border: 1px solid #27272a;
    border-radius: 12px;
    overflow: hidden;
  }
  .alllogs-head {
    display: flex; align-items: center; gap: 10px;
    padding: 14px 18px; cursor: pointer; user-select: none;
  }
  .alllogs-head:hover { background: #1f2028; }
  .alllogs-title { font-weight: 700; font-size: 0.9em; }
  .alllogs-count { color: #52525b; font-size: 0.78em; }
  .alllogs-chev { margin-left: auto; color: #52525b; font-size: 0.9em; }
  .alllogs-body {
    border-top: 1px solid #27272a;
    max-height: 520px; overflow-y: auto;
    padding: 14px 18px;
  }
  .alllogs-body.collapsed { display: none; }
  .alllog-sec { margin-bottom: 16px; }
  .alllog-sec:last-child { margin-bottom: 4px; }
  .alllog-head { display: flex; align-items: center; gap: 8px; margin-bottom: 6px; font-size: 0.82em; }
  .alllog-name { font-weight: 600; }
  .alllog-port { color: #52525b; font-family: monospace; font-size: 0.9em; }
  .alllog-pre {
    background: #0f1117; border: 1px solid #27272a; border-radius: 8px;
    padding: 10px; margin: 0;
    font-family: 'JetBrains Mono', 'Fira Code', monospace;
    font-size: 0.72em; color: #9ca3af;
    max-height: 220px; overflow: auto;
    white-space: pre-wrap; word-break: break-all;
    user-select: text; -webkit-user-select: text;
  }
  .row-logbox-wrap { display: flex; flex-direction: column; }
  .row-logbox-header {
    display: flex; align-items: center; justify-content: space-between;
    padding: 4px 0; margin-bottom: 4px; font-size: 0.75em;
    color: #71717a; font-weight: 600;
  }
  .row-logbox-header span { white-space: nowrap; }
  .row-logbox { user-select: text; -webkit-user-select: text; }
  .copy-btn {
    background: #27272a; border: 1px solid #3f3f46; border-radius: 6px;
    color: #a1a1aa; padding: 3px 10px; font-size: 0.75em; cursor: pointer;
    font-family: inherit; transition: all 0.15s;
  }
  .copy-btn:hover { background: #3f3f46; color: #e4e4e7; }
  .copy-btn.copied { background: #22c55e; color: #fff; border-color: #22c55e; }
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
    </div>
    <div class="analytics-charts">
      <div class="analytics-chart-box">
        <div style="display:flex;align-items:center;justify-content:space-between;margin-bottom:10px">
          <div class="analytics-chart-title" style="margin-bottom:0" id="chart-daily-title">Tokens per Day</div>
          <div style="display:flex;gap:3px">
            <button class="range-btn active" id="toggle-tokens" onclick="setChartMode('tokens')">Tokens</button>
            <button class="range-btn" id="toggle-cost" onclick="setChartMode('cost')">$ Cost</button>
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

<div class="table-bar">
  <input type="search" class="search-input" id="search" placeholder="Search models, tags, ports, quant…">
  <span class="table-count" id="table-count"></span>
</div>

<div class="table-wrap">
  <div class="table-card">
    <table id="model-table">
      <thead>
        <tr>
          <th data-key="name">Name</th>
          <th data-key="quant">Quant</th>
          <th data-key="vram">VRAM Usage</th>
          <th data-key="port">Port</th>
          <th data-key="tags">Tags</th>
          <th data-key="type">Type / Description</th>
          <th data-key="offload">CPU Offload</th>
          <th data-key="logs">Show Logs</th>
          <th data-key="run">Start/Stop</th>
          <th data-key="fav">Favorite</th>
          <th data-key="status">Status</th>
        </tr>
      </thead>
      <tbody id="model-tbody"></tbody>
    </table>
  </div>
</div>

<div class="alllogs">
  <div class="alllogs-panel">
    <div class="alllogs-head" onclick="toggleAllLogs()">
      <span class="alllogs-title">All Model Logs</span>
      <span class="alllogs-count" id="alllogs-count">collapsed</span>
      <span class="alllogs-chev" id="alllogs-chev">▸</span>
    </div>
    <div class="alllogs-body collapsed" id="alllogs-body"></div>
  </div>
</div>

<div class="footer">Load &middot; Use &middot; Unload</div>

<script src="https://cdn.jsdelivr.net/npm/chart.js@4.4.0/dist/chart.umd.min.js"></script>
<script>
const HOST = location.hostname;
const tbody = document.getElementById('model-tbody');
let currentModels = {};
let sortKey = 'fav';
let sortDir = -1;
let searchQuery = '';
let rowMap = {};
let logIntervals = {};
let favorites = new Set();
try { favorites = new Set(JSON.parse(localStorage.getItem('fav_models') || '[]')); } catch (e) {}

function saveFavs() {
  try { localStorage.setItem('fav_models', JSON.stringify([...favorites])); } catch (e) {}
}

function toggleFav(id) {
  if (favorites.has(id)) favorites.delete(id); else favorites.add(id);
  saveFavs();
  const star = document.getElementById('star-' + id);
  if (star) star.classList.toggle('on', favorites.has(id));
  if (sortKey === 'fav') renderAll(currentModels);
}

function sortValue(id, m, key) {
  switch (key) {
    case 'name': return m.name.toLowerCase();
    case 'quant': return (m.quant || '').toLowerCase() || '~';
    case 'vram': return m.vram_gb == null ? -1 : m.vram_gb;
    case 'port': return m.port;
    case 'tags': return (m.tags || []).join(', ').toLowerCase();
    case 'type': return (m.category || '') + ' :: ' + m.name.toLowerCase();
    case 'offload': return m.supports_offload ? 1 : 0;
    case 'logs': return rowMap[id] && rowMap[id].logOpen ? 1 : 0;
    case 'run': return m.status === 'stopped' ? 0 : 1;
    case 'fav': return favorites.has(id) ? 1 : 0;
    case 'status': { const o = { ready: 0, starting: 1, stopped: 2 }; return o[m.status] != null ? o[m.status] : 3; }
  }
  return '';
}

function sortRows() {
  const ids = Object.keys(currentModels);
  ids.sort((a, b) => {
    const va = sortValue(a, currentModels[a], sortKey);
    const vb = sortValue(b, currentModels[b], sortKey);
    let r = va < vb ? -1 : (va > vb ? 1 : 0);
    if (r === 0) return currentModels[a].name.localeCompare(currentModels[b].name);
    return r * sortDir;
  });
  return ids;
}

function matchesSearch(m) {
  if (!searchQuery) return true;
  const q = searchQuery.toLowerCase();
  const hay = [m.name, m.description || '', m.category || '', String(m.port), (m.tags || []).join(' '), m.quant || ''].join(' ').toLowerCase();
  return hay.includes(q);
}

document.querySelectorAll('#model-table th[data-key]').forEach(th => {
  th.addEventListener('click', () => {
    const key = th.dataset.key;
    if (sortKey === key) sortDir *= -1; else { sortKey = key; sortDir = 1; }
    document.querySelectorAll('#model-table th[data-key]').forEach(h => h.classList.remove('sorted-asc', 'sorted-desc'));
    th.classList.add(sortDir === 1 ? 'sorted-asc' : 'sorted-desc');
    renderAll(currentModels);
  });
});

document.getElementById('search').addEventListener('input', e => {
  searchQuery = e.target.value.trim();
  renderAll(currentModels);
});

document.querySelector('#model-table th[data-key="fav"]').classList.add('sorted-desc');

function buildRow(id, m) {
  const tr = document.createElement('tr');
  tr.className = 'model-row';
  const initial = (m.name || '?').trim().charAt(0);
  tr.innerHTML = `
    <td class="td-name"><span class="row-icon" style="background:${m.color}18;color:${m.color}">${initial}</span><a class="name-link" id="link-${id}" target="_blank" rel="noopener">${m.name}</a></td>
    <td class="mono">${m.quant || '—'}</td>
    <td class="mono">${m.vram_gb ? '~' + m.vram_gb + ' GB' : '—'}</td>
    <td class="mono">${m.port}</td>
    <td class="td-tags">${(m.tags || []).map(t => `<span class="tag-badge">${t}</span>`).join('')}</td>
    <td class="td-desc"><span class="cat-badge" style="background:${m.color}18;color:${m.color}">${m.category}</span><span class="desc-text">${m.description}</span></td>
    <td>${m.supports_offload ? `<input type="checkbox" class="switch" id="offload-${id}">` : '<span class="dim">—</span>'}</td>
    <td><input type="checkbox" class="switch" onchange="toggleRowLogs('${id}', this.checked)"></td>
    <td><input type="checkbox" class="switch" id="runsw-${id}" onchange="toggleRun('${id}', this.checked)" disabled></td>
    <td><button class="star" id="star-${id}" title="favorite" onclick="toggleFav('${id}')">&#9733;</button></td>
    <td class="td-status"><span class="status-dot" id="dot-${id}"></span><span class="status-text" id="statustext-${id}"></span></td>`;
  const logTr = document.createElement('tr');
  logTr.className = 'log-row';
  logTr.style.display = 'none';
  logTr.innerHTML = `<td colspan="11"><div class="row-logbox" id="rowlog-${id}"></div></td>`;
  rowMap[id] = { tr: tr, logTr: logTr, logOpen: false };
  tbody.appendChild(tr);
  tbody.appendChild(logTr);
}

function updateRow(id, m) {
  const dot = document.getElementById('dot-' + id);
  const st = document.getElementById('statustext-' + id);
  if (dot) dot.className = 'status-dot ' + m.status;
  if (st) st.textContent = m.status === 'ready' ? 'Running' : m.status === 'starting' ? 'Starting…' : 'Stopped';
  const link = document.getElementById('link-' + id);
  if (link) {
    if (m.status === 'ready') {
      link.href = (m.protocol || 'http') + '://' + HOST + ':' + m.port + (m.path || '');
      link.classList.add('active');
    } else {
      link.removeAttribute('href');
      link.classList.remove('active');
    }
  }
  const runsw = document.getElementById('runsw-' + id);
  if (runsw) { runsw.checked = m.status !== 'stopped'; runsw.disabled = m.status === 'starting'; }
  const off = document.getElementById('offload-' + id);
  if (off) off.disabled = m.status !== 'stopped';
  const star = document.getElementById('star-' + id);
  if (star) star.classList.toggle('on', favorites.has(id));
}

function renderAll(models) {
  currentModels = models;
  let runCount = 0;
  for (const [id, m] of Object.entries(models)) {
    if (!rowMap[id]) buildRow(id, m);
    if (m.status !== 'stopped') runCount++;
    updateRow(id, m);
  }
  document.getElementById('running-count').textContent = runCount + ' model' + (runCount !== 1 ? 's' : '') + ' running';

  for (const id of sortRows()) {
    tbody.appendChild(rowMap[id].tr);
    tbody.appendChild(rowMap[id].logTr);
  }

  let visible = 0;
  for (const [id, m] of Object.entries(models)) {
    const show = matchesSearch(m);
    if (show) visible++;
    rowMap[id].tr.style.display = show ? '' : 'none';
    rowMap[id].logTr.style.display = (show && rowMap[id].logOpen) ? '' : 'none';
  }
  document.getElementById('table-count').textContent = visible + ' / ' + Object.keys(models).length + ' models';
  document.getElementById('alllogs-count').textContent = allLogsOpen ? runCount + ' model' + (runCount !== 1 ? 's' : '') + ' streaming' : 'collapsed';
}

async function refresh() {
  try {
    const r = await fetch('/api/status');
    renderAll(await r.json());
  } catch (e) {}
}

async function toggleRun(id, on) {
  if (on) {
    const cb = document.getElementById('offload-' + id);
    const offload = cb && cb.checked ? 'true' : 'false';
    await fetch('/api/start/' + id + '?offload=' + offload, { method: 'POST' });
  } else {
    await fetch('/api/stop/' + id, { method: 'POST' });
  }
  setTimeout(refresh, 600);
}

async function fetchRowLog(id) {
  const e = rowMap[id];
  if (!e || !e.logOpen) return;
  const box = document.getElementById('rowlog-' + id);
  if (!box) return;
  try {
    const wasAtBottom = box.scrollHeight - box.scrollTop - box.clientHeight < 40;
    const r = await fetch('/api/logs/' + id);
    const d = await r.json();
    box.textContent = d.logs || '(no logs yet)';
    if (wasAtBottom) box.scrollTop = box.scrollHeight;
  } catch (err) {
    box.textContent = '(error loading logs)';
  }
}

function toggleRowLogs(id, open) {
  const e = rowMap[id];
  if (!e) return;
  e.logOpen = open;
  e.logTr.style.display = open ? '' : 'none';
  if (open) {
    fetchRowLog(id);
    if (!logIntervals[id]) logIntervals[id] = setInterval(() => fetchRowLog(id), 3000);
  } else if (logIntervals[id]) {
    clearInterval(logIntervals[id]);
    delete logIntervals[id];
  }
}

let allLogsOpen = false;

function toggleAllLogs() {
  allLogsOpen = !allLogsOpen;
  document.getElementById('alllogs-body').classList.toggle('collapsed', !allLogsOpen);
  document.getElementById('alllogs-chev').textContent = allLogsOpen ? '\u25be' : '\u25b8';
  document.getElementById('alllogs-count').textContent = allLogsOpen ? 'streaming…' : 'collapsed';
  if (allLogsOpen) updateAllLogs();
}

function runningIds() {
  return Object.keys(currentModels).filter(id => currentModels[id].status !== 'stopped');
}

async function updateAllLogs() {
  const ids = runningIds();
  if (!allLogsOpen) return;
  const body = document.getElementById('alllogs-body');
  for (const id of ids) {
    const m = currentModels[id];
    let pre = document.getElementById('alllogpre-' + id);
    if (!pre) {
      const sec = document.createElement('div');
      sec.className = 'alllog-sec';
      sec.dataset.id = id;
      sec.innerHTML = `<div class="alllog-head"><span class="status-dot ${m.status}"></span><span class="alllog-name">${m.name}</span><span class="alllog-port">${m.protocol || 'http'}://${HOST}:${m.port}</span></div><pre class="alllog-pre" id="alllogpre-${id}"></pre>`;
      body.appendChild(sec);
      pre = document.getElementById('alllogpre-' + id);
    }
    (async () => {
      try {
        const wasAtBottom = pre.scrollHeight - pre.scrollTop - pre.clientHeight < 40;
        const r = await fetch('/api/logs/' + id);
        const d = await r.json();
        pre.textContent = d.logs || '(no logs yet)';
        if (wasAtBottom) pre.scrollTop = pre.scrollHeight;
      } catch (e) {}
    })();
  }
  for (const sec of Array.from(body.querySelectorAll('.alllog-sec'))) {
    if (!ids.includes(sec.dataset.id)) sec.remove();
  }
}

// ── Token analytics ──────────────────────────────────────────────
let tokenData = null;
let powerByDay = {};
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
    updateCost();
  } catch(e) {}
}

function setChartMode(mode) {
  chartMode = mode;
  document.getElementById('toggle-tokens').classList.toggle('active', mode === 'tokens');
  document.getElementById('toggle-cost').classList.toggle('active', mode === 'cost');
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
  } else {
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
setInterval(() => { if (allLogsOpen) updateAllLogs(); }, 3000);
</script>
</body>
</html>
"""

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=80)
