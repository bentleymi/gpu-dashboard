# GPU Model Dashboard

A single-file FastAPI dashboard for managing and monitoring AI models running on NVIDIA GPUs.

## Features

- **Model lifecycle management** — Start, stop, and monitor 30+ AI models (LLMs, image generation, video, audio)
- **Real-time GPU monitoring** — VRAM usage, core utilization, temperature, and power draw
- **Token analytics** — Pulls usage data from the OpenCode SQLite database, broken down by model and day
- **Power cost tracking** — Samples GPU power via `nvidia-smi` every 30s, accumulates watt-hours per day
- **Historical backfill** — Estimates power consumption for past days using token counts from the OpenCode DB and model-specific power profiles
- **Interactive charts** — Time-range selector (7d / 30d / 90d / 1y / All) with token, cost, and power (kWh) views

## Quick Start

```bash
cd /mnt/raid1_sata/JanusPro7b
./dashboard-venv/bin/python dashboard.py
```

The dashboard serves on `http://0.0.0.0:80` (port 80).

### With systemd (recommended)

A systemd service file can be created to keep the dashboard running:

```ini
[Unit]
Description=GPU Dashboard
After=network.target

[Service]
ExecStart=/mnt/raid1_sata/JanusPro7b/dashboard-venv/bin/python /mnt/raid1_sata/JanusPro7b/dashboard.py
WorkingDirectory=/mnt/raid1_sata/JanusPro7b
Restart=always

[Install]
WantedBy=multi-user.target
```

```bash
sudo cp gpu-dashboard.service /etc/systemd/system/
sudo systemctl daemon-reload
sudo systemctl enable --now gpu-dashboard.service
```

## Configuration

### Models

Models are defined in the `MODELS` dictionary at the top of `dashboard.py`. Each entry specifies:

| Field | Description |
|-------|-------------|
| `name` | Display name |
| `port` | HTTP port the model serves on |
| `cmd` | Command to start the model |
| `cwd` | Working directory |
| `category` | Grouping: `Image`, `Video`, `Audio`, `LLM`, `Tools` |
| `systemd_service` | (optional) If set, start/stop delegates to systemctl |

### Power Profiles

Model-specific power profiles are defined in `MODEL_POWER_PROFILES` (line ~630). Each model type has:

- `watts` — Power draw when actively processing
- `idle_watts` — Baseline GPU power when that model is loaded but idle

These are used for historical backfill when real power samples are unavailable. Edit these values if your hardware draws differently.

### kWh Rate

The cost per kWh is configurable in the dashboard UI (top-right input field). The default is `$0.12/kWh`.

## Power & Cost Estimation

### Real data
The dashboard samples GPU power draw via `nvidia-smi` every 30 seconds and accumulates watt-hours per calendar day. This data is stored in `logs/power_usage.json`.

### Backfill
For days without real power samples, the dashboard estimates power using:
1. Token counts from the OpenCode database (`/root/.local/share/opencode/opencode.db`)
2. Model-specific power profiles (watts, tokens/min throughput)
3. Active-hours estimation based on token volume

This provides a reasonable approximation but should be treated as an estimate, not a measurement.

## API Endpoints

| Endpoint | Description |
|----------|-------------|
| `GET /api/status` | Status of all managed models |
| `GET /api/gpu` | Real-time GPU metrics (VRAM, temp, power) |
| `GET /api/power` | Historical power consumption (watt-hours per day) |
| `GET /api/tokens?days=N` | Token usage stats from OpenCode DB |
| `POST /api/start/{model_id}` | Start a model (optionally with `?offload=true`) |
| `POST /api/stop/{model_id}` | Stop a model |
| `GET /api/logs/{model_id}` | Tail model logs |

## File Structure

```
dashboard.py          # Single-file FastAPI app (dashboard + all logic)
logs/power_usage.json # Accumulated watt-hours per day (real samples)
logs/power_history.csv # CSV log of power samples for backfill reference
```

## Hardware

Tested on NVIDIA RTX PRO 6000 Blackwell Workstation Edition (96 GB VRAM). Adjust power profiles in `MODEL_POWER_PROFILES` for different hardware.