# WeaveHacks3 — Self‑Improving Agent (Marimo + Redis + Weave)

## Overview
- The Marimo app [WeaveHacks3/marimo_agentic_b.py](WeaveHacks3/marimo_agentic_b.py) predicts collision risk from a video path.
- Each prediction builds an evaluation and can mirror it to Redis Streams.
- A calibration file [WeaveHacks3/collision_calibration.json](WeaveHacks3/collision_calibration.json) is updated from Redis evals to refine bins and threshold.

## Install
```bash
pip install -r WeaveHacks3/requirements.txt
```

## Enable Redis + Calibration
```powershell
# Redis: eval mirroring (Streams)
$env:REDIS_EVALS_ENABLE = "1"
$env:REDIS_EVALS_URL = "redis://localhost:6379/0"
$env:REDIS_EVALS_STREAM_KEY = "evals:events"

# Calibration from Redis (bootstrap creates file even with few/no evals)
$env:REDIS_CALIBRATE_ENABLE = "1"
$env:CALIBRATION_BOOTSTRAP = "1"
$env:CALIBRATION_BIN_COUNT = "10"     # optional
$env:CALIBRATION_MIN_RECORDS = "5"    # optional

# Optional: local JSONL collection + mirroring for retrain
$env:REDIS_ENABLE = "1"                # for collector stream/list
$env:REDIS_URL = "redis://localhost:6379/0"
$env:REDIS_STREAM_KEY = "collision:events"
$env:REDIS_LOG_ALL = "1"               # log all predictions to Redis
$env:COLLECT_LOWER = "0.4"; $env:COLLECT_UPPER = "0.6"
```

## Run the Agent (Marimo)
```powershell
python WeaveHacks3\marimo_agentic_b.py
```
- In the chat, ask it to analyze a real video path (e.g., "Analyze C:\\videos\\sample.mp4").
- On each prediction: an eval is sent, mirrored to Redis, and calibration updates [WeaveHacks3/collision_calibration.json](WeaveHacks3/collision_calibration.json).

## Emit Evals Quickly (CLI)
[WeaveHacks3/run_eval.py](WeaveHacks3/run_eval.py) now emits per‑row evaluations via `send_evaluation()`.

Mock (no videos):
```powershell
$env:EMIT_ONLY = "1"
python WeaveHacks3\run_eval.py jsonl
```

Agent rows (uses your model; falls back when file missing):
```powershell
$env:EMIT_ONLY = "1"
python WeaveHacks3\run_eval.py agent
```

Unset `EMIT_ONLY` to also run the full Weave evaluation.

## Inspect Redis
```powershell
python WeaveHacks3\redis_consumer.py --mode stream --key evals:events --start 0-0 --max 5
```

## Redis Analysis
- Quick stats from the stream using the built-in analyzer:
```powershell
# Populate some evals first (see Emit Evals Quickly)
$env:REDIS_EVALS_URL = "redis://localhost:6379/0"
python WeaveHacks3\redis_analysis.py --stream-key evals:events --max-read 1000 --bins 10 --threshold 0.75
```
- Example output: total evals, positive rate, average probability, bin counts, label coverage, accuracy/F1 when labels exist.
- Filtered tail views:
```powershell
# Only collision_detection tasks
python WeaveHacks3\redis_consumer.py --mode stream --key evals:events --start 0-0 --task collision_detection --max 50
# Only positive decisions
python WeaveHacks3\redis_consumer.py --mode stream --key evals:events --start 0-0 --filter decision=true --max 50
```

## Verify Calibration
```powershell
Get-Content WeaveHacks3\collision_calibration.json
```

## Troubleshooting
- Calibration file missing: ensure `$env:REDIS_CALIBRATE_ENABLE = "1"` and `$env:CALIBRATION_BOOTSTRAP = "1"`, then run a prediction.
- No evals in Redis: export `REDIS_EVALS_ENABLE=1` and emit via `run_eval.py` or the Marimo app.
- `redis-cli` not found: use the Python consumer [WeaveHacks3/redis_consumer.py](WeaveHacks3/redis_consumer.py).
	For stream length and raw XRANGE, you can also run (if redis-cli is installed):
```powershell
redis-cli XLEN evals:events
redis-cli XRANGE evals:events - + COUNT 10
```

## Reference — Collector Env Vars
- `REDIS_ENABLE`: enable Redis for the JSONL collector.
- `REDIS_URL` or `REDIS_HOST`/`REDIS_PORT`/`REDIS_DB`: connection.
- `REDIS_STREAM_KEY` / `REDIS_LIST_KEY`: destination key.
- `REDIS_STREAM_MAXLEN` / `REDIS_LIST_MAXLEN`: retention caps.
- `ALWAYS_COLLECT`, `COLLECT_LOWER`, `COLLECT_UPPER`: JSONL collection policy.
- `REDIS_LOG_ALL`: mirror all predictions to Redis.

## Reference — Evals & Calibration Env Vars
- `REDIS_EVALS_ENABLE`, `REDIS_EVALS_URL`, `REDIS_EVALS_STREAM_KEY`: mirror evals to streams.
- `REDIS_CALIBRATE_ENABLE`: enable Redis‑driven calibration.
- `CALIBRATION_BOOTSTRAP`, `CALIBRATION_MIN_RECORDS`, `CALIBRATION_MAX_READ`, `CALIBRATION_BIN_COUNT`.

## Notes
- Labels improve threshold learning (`reference.label`). Without labels, bin rates use decisions; threshold falls back to prior value.
- Retrain trigger exists in the collector via `TRIGGER_AFTER` and `RETRAIN_SCRIPT`.
- For windows users wanting to use Redis commands, they may need to install WSL first and then install redis-cli via sudo.