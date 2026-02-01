import marimo
mo = marimo

__generated_with = "0.19.7"
app = marimo.App()

@app.cell
def _import_mo():
    import marimo as mo
    return (mo,)

@app.cell
def _collision_agent():
    import ell
    import marimo
    from marimofiguration import CollisionVideoModel, CollisionDetectionAgent
    from openai import OpenAI
    # evaluation helpers
    from evals import build_evaluation, compute_metrics_for_prediction, send_evaluation
    import os as _os
    import weave
    import json as _json
    import time
    import pathlib
    import subprocess
    import sys
    import logging
    # optional Redis logging
    # (import inside cell to avoid global dependency failures)
    # self-improving calibration from Redis evals

    client = weave.init(_os.getenv("WEAVE_PROJECT", "WeaveHacks3-Evals"))
    client.add_cost(
        llm_id="gpt-4o",
        prompt_token_cost=0.03 / 1000,
        completion_token_cost=0.06 / 1000,
    )

    @ell.tool()
    def collision_predictor(path : str) -> float:
        """Predict collision probability from video file at given path."""
        model = CollisionVideoModel()
        # Default threshold; may be overridden by calibration if available
        default_threshold = 0.75
        agent = CollisionDetectionAgent(model, threshold=default_threshold)

        if not _os.path.exists(path):
            raise FileNotFoundError(f"Houston, No filepath here, over... (File '{path}' does not exist.)")

        input_tensor = agent.perceive(path)
        prob = agent.think(input_tensor)

        # Optional: load calibrator for better decisioning
        cal_path = _os.path.join(_os.path.dirname(__file__), "collision_calibration.json")
        calibrated_prob = prob
        used_threshold = default_threshold
        try:
            if _os.path.exists(cal_path):
                with open(cal_path, "r", encoding="utf-8") as fh:
                    cal = _json.load(fh)
                bins = cal.get("bins", [])
                # map prob to calibrated rate via bins
                for b in bins:
                    low = float(b.get("low", 0.0))
                    high = float(b.get("high", 1.0))
                    rate = b.get("rate", None)
                    if rate is None:
                        continue
                    if low <= prob <= high:
                        calibrated_prob = float(rate)
                        break
                used_threshold = float(cal.get("threshold", default_threshold))
        except Exception:
            # if calibration fails, fall back silently
            pass

        decision = calibrated_prob > used_threshold

        # Build and send an evaluation record for this prediction
        try:
            eval_payload = build_evaluation(
                model_version=_os.getenv("MODEL_VERSION", "v1.0"),
                task="collision_detection",
                input_obj={"video_path": path},
                prediction=calibrated_prob,
                decision=decision,
                session_id=_os.getenv("SESSION_ID", None),
                metadata={
                    "frame_count": getattr(input_tensor, "frame_count", None),
                    "raw_prob": float(prob),
                    "calibrated": _os.path.exists(cal_path),
                    "threshold": used_threshold,
                },
            )
            eval_payload["metrics"] = compute_metrics_for_prediction(calibrated_prob, threshold=used_threshold)
            # send evaluation via weave/evals SDK when possible
            try:
                ingest_url = _os.getenv("WEAVE_INGEST_URL")
                ingest_headers = None
                send_evaluation(client=client, evaluation=eval_payload, ingest_url=ingest_url, ingest_headers=ingest_headers)
            except Exception:
                # fallback: log locally and continue
                logging.warning("send_evaluation failed; continuing and saving local record")
            # Optional self-improvement: rebuild calibration from Redis evals
            try:
                _maybe_update_calibration_from_redis_evals(cal_path)
            except Exception:
                # non-fatal
                pass
        except Exception:
            # don't break prediction on evaluation failures
            logging.warning("Failed to build evaluation payload")

        # --- local collection for retraining / labeling ---
        try:
            # config
            collect_path = pathlib.Path(_os.getenv("COLLECT_PATH", "data/collected.jsonl"))
            collect_path.parent.mkdir(parents=True, exist_ok=True)
            collect_lower = float(_os.getenv("COLLECT_LOWER", "0.4"))
            collect_upper = float(_os.getenv("COLLECT_UPPER", "0.6"))
            always_collect = _os.getenv("ALWAYS_COLLECT", "0") == "1"
            trigger_after = int(_os.getenv("TRIGGER_AFTER", "0"))
            retrain_script = _os.getenv("RETRAIN_SCRIPT", _os.path.join(_os.path.dirname(__file__), "retrain.py"))
            retrain_epochs = _os.getenv("RETRAIN_EPOCHS", "5")

            # Optional: Redis wiring via env vars
            # REDIS_ENABLE=1 to enable; provide REDIS_URL or host/port
            redis_enable = _os.getenv("REDIS_ENABLE", "0") == "1"
            redis_url = _os.getenv("REDIS_URL", None)
            redis_stream_key = _os.getenv("REDIS_STREAM_KEY", "collision:events")
            redis_list_key = _os.getenv("REDIS_LIST_KEY", None)
            redis_stream_maxlen = int(_os.getenv("REDIS_STREAM_MAXLEN", "10000"))
            redis_list_maxlen = _os.getenv("REDIS_LIST_MAXLEN", None)
            redis_log_all = _os.getenv("REDIS_LOG_ALL", "0") == "1"
            _r = None
            if redis_enable:
                try:
                    import redis as _redis
                    if redis_url:
                        _r = _redis.from_url(redis_url)
                    else:
                        _r = _redis.Redis(
                            host=_os.getenv("REDIS_HOST", "localhost"),
                            port=int(_os.getenv("REDIS_PORT", "6379")),
                            db=int(_os.getenv("REDIS_DB", "0")),
                        )
                except Exception as _re:
                    logging.warning("Redis init failed: %s", _re)
                    _r = None

            def append_record(rec: dict):
                with collect_path.open("a", encoding="utf-8") as fh:
                    fh.write(_json.dumps(rec, default=str) + "\n")

            def append_to_redis(rec: dict):
                if not _r:
                    return
                try:
                    payload = _json.dumps(rec, default=str)
                    if redis_stream_key:
                        # Store JSONL as a stream entry with helpful fields
                        fields = {
                            "json": payload,
                            "video_path": rec.get("video_path", ""),
                            "model_version": rec.get("model_version", ""),
                            "decision": str(rec.get("decision", "")),
                        }
                        _r.xadd(
                            redis_stream_key,
                            fields,
                            maxlen=redis_stream_maxlen,
                            approximate=True,
                        )
                    elif redis_list_key:
                        _r.lpush(redis_list_key, payload)
                        if redis_list_maxlen is not None:
                            try:
                                _r.ltrim(redis_list_key, 0, int(redis_list_maxlen) - 1)
                            except Exception:
                                pass
                except Exception as e:
                    logging.warning("Failed to append to Redis: %s", e)

            def maybe_trigger_retrain():
                if trigger_after <= 0:
                    return
                try:
                    n = sum(1 for _ in collect_path.open("r", encoding="utf-8"))
                except FileNotFoundError:
                    n = 0
                if n > 0 and n % trigger_after == 0:
                    cmd = [sys.executable, retrain_script, "--input", str(collect_path), "--epochs", retrain_epochs]
                    # spawn detached
                    subprocess.Popen(cmd, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)

            record = {
                "timestamp": time.time(),
                "video_path": path,
                "prediction": float(prob),
                "decision": bool(decision),
                "model_version": _os.getenv("MODEL_VERSION", "v1.0"),
                "metadata": {
                    "frame_count": getattr(input_tensor, "frame_count", None),
                    "calibrated_prob": float(calibrated_prob),
                    "threshold": used_threshold,
                },
            }

            should_collect_jsonl = always_collect or (collect_lower <= prob <= collect_upper)
            if should_collect_jsonl:
                append_record(record)
                maybe_trigger_retrain()
            # Log to Redis if enabled: mirror JSONL collection or log all if requested
            if should_collect_jsonl or redis_log_all:
                append_to_redis(record)
        except Exception as e:
            logging.warning("Failed to write local collect record: %s", e)

        agent.act(decision, calibrated_prob, path)
        return calibrated_prob

    def _maybe_update_calibration_from_redis_evals(save_path: str):
        """Build/update collision calibration file based on Redis evals stream.

        Controlled via env:
          - REDIS_CALIBRATE_ENABLE=1 to enable
          - REDIS_EVALS_URL or REDIS_HOST/REDIS_PORT/REDIS_DB for connection
          - REDIS_EVALS_STREAM_KEY (default 'evals:events')
          - CALIBRATION_MIN_RECORDS (default '50')
          - CALIBRATION_MAX_READ (default '1000')
          - CALIBRATION_BIN_COUNT (default '10')
        """
        if _os.getenv("REDIS_CALIBRATE_ENABLE", "0") != "1":
            return
        try:
            import redis as _redis
        except Exception:
            # If bootstrap is enabled, write a default calibration file even without Redis
            if bootstrap:
                try:
                    step = 1.0 / bin_count
                    bins = [
                        {
                            "low": i * step,
                            "high": (i + 1) * step if i < bin_count - 1 else 1.0,
                            "rate": None,
                        }
                        for i in range(bin_count)
                    ]
                    out = {
                        "bins": bins,
                        "threshold": 0.75,
                        "updated_ts": time.time(),
                        "source": "bootstrap_no_redis",
                    }
                    pathlib.Path(save_path).parent.mkdir(parents=True, exist_ok=True)
                    with open(save_path, "w", encoding="utf-8") as fh:
                        fh.write(_json.dumps(out, indent=2))
                    logging.info("Calibration: bootstrapped default file '%s' (no Redis)", save_path)
                except Exception as e:
                    logging.warning("Calibration: bootstrap write failed '%s': %s", save_path, e)
            return
        url = _os.getenv("REDIS_EVALS_URL")
        if url:
            r = _redis.from_url(url)
        else:
            r = _redis.Redis(
                host=_os.getenv("REDIS_HOST", "localhost"),
                port=int(_os.getenv("REDIS_PORT", "6379")),
                db=int(_os.getenv("REDIS_DB", "0")),
            )
        stream_key = _os.getenv("REDIS_EVALS_STREAM_KEY", "evals:events")
        max_read = int(_os.getenv("CALIBRATION_MAX_READ", "1000"))
        bin_count = max(2, int(_os.getenv("CALIBRATION_BIN_COUNT", "10")))
        min_records = int(_os.getenv("CALIBRATION_MIN_RECORDS", "50"))
        bootstrap = _os.getenv("CALIBRATION_BOOTSTRAP", "0") == "1"

        try:
            entries = r.xrevrange(stream_key, "+", "-", count=max_read)
        except Exception as e:
            logging.warning("Calibration: failed to read Redis stream '%s': %s", stream_key, e)
            # If bootstrap is enabled, still write a default calibration file
            if bootstrap:
                try:
                    step = 1.0 / bin_count
                    bins = [
                        {
                            "low": i * step,
                            "high": (i + 1) * step if i < bin_count - 1 else 1.0,
                            "rate": None,
                        }
                        for i in range(bin_count)
                    ]
                    out = {
                        "bins": bins,
                        "threshold": 0.75,
                        "updated_ts": time.time(),
                        "source": "bootstrap_stream_error",
                    }
                    pathlib.Path(save_path).parent.mkdir(parents=True, exist_ok=True)
                    with open(save_path, "w", encoding="utf-8") as fh:
                        fh.write(_json.dumps(out, indent=2))
                    logging.info("Calibration: bootstrapped default file '%s' (stream error)", save_path)
                except Exception as e2:
                    logging.warning("Calibration: bootstrap write failed '%s': %s", save_path, e2)
            return

        probs = []
        labels = []
        decisions = []

        def _dec(v):
            if isinstance(v, bytes):
                try:
                    return v.decode("utf-8", errors="replace")
                except Exception:
                    return v
            return v

        for _, fields in entries:
            # unpack json payload
            j = fields.get("json")
            j = _dec(j)
            if not j:
                continue
            try:
                ev = _json.loads(j)
            except Exception:
                continue
            try:
                pr = float(((ev.get("prediction") or {}).get("probability")))
            except Exception:
                continue
            probs.append(pr)
            # collect decision
            try:
                decisions.append(bool(((ev.get("prediction") or {}).get("decision"))))
            except Exception:
                decisions.append(None)
            # collect label if present
            ref = ev.get("reference") or {}
            lab = ref.get("label", None)
            if lab is not None:
                try:
                    labels.append(bool(int(lab)))
                except Exception:
                    try:
                        labels.append(bool(lab))
                    except Exception:
                        labels.append(None)
            else:
                labels.append(None)

        # Need enough records, unless bootstrapping enabled
        if len(probs) < min_records and not bootstrap:
            logging.info("Calibration: insufficient records (%d < %d); skipping", len(probs), min_records)
            return

        # Build equal-width bins [0,1]
        step = 1.0 / bin_count
        bins = []
        for i in range(bin_count):
            low = i * step
            high = (i + 1) * step if i < bin_count - 1 else 1.0
            idxs = [k for k, p in enumerate(probs) if low <= p <= high]
            # estimate rate using labels when available; fallback to decisions
            vals = [labels[k] for k in idxs if labels[k] is not None]
            if len(vals) >= 3:  # at least a few labels in bin
                rate = sum(1 for v in vals if v) / float(len(vals))
            else:
                dvals = [decisions[k] for k in idxs if decisions[k] is not None]
                rate = (sum(1 for v in dvals if v) / float(len(dvals))) if dvals else None
            bins.append({"low": low, "high": high, "rate": rate})

        # Choose threshold: optimize F1 if labels available, else keep existing
        chosen_threshold = None
        if any(v is not None for v in labels):
            # simple sweep
            best_f1 = -1.0
            best_t = 0.75
            for t_i in [i / 100.0 for i in range(10, 100, 1)]:
                tp = fp = fn = 0
                for p, lab in zip(probs, labels):
                    if lab is None:
                        continue
                    pred_pos = p > t_i
                    if pred_pos and lab:
                        tp += 1
                    elif pred_pos and not lab:
                        fp += 1
                    elif (not pred_pos) and lab:
                        fn += 1
                prec = tp / float(tp + fp) if (tp + fp) > 0 else 0.0
                rec = tp / float(tp + fn) if (tp + fn) > 0 else 0.0
                f1 = (2 * prec * rec) / float(prec + rec) if (prec + rec) > 0 else 0.0
                if f1 > best_f1:
                    best_f1 = f1
                    best_t = t_i
            chosen_threshold = best_t

        # Load existing calibration to merge
        existing = None
        try:
            if _os.path.exists(save_path):
                with open(save_path, "r", encoding="utf-8") as fh:
                    existing = _json.load(fh)
        except Exception:
            existing = None

        out = {
            "bins": bins,
            "threshold": float(chosen_threshold if chosen_threshold is not None else existing.get("threshold", 0.75) if isinstance(existing, dict) else 0.75),
            "updated_ts": time.time(),
            "source": "redis_evals",
        }
        try:
            pathlib.Path(save_path).parent.mkdir(parents=True, exist_ok=True)
            with open(save_path, "w", encoding="utf-8") as fh:
                fh.write(_json.dumps(out, indent=2))
            logging.info("Calibration: wrote file '%s' with %d bins", save_path, len(bins))
        except Exception as e:
            logging.warning("Calibration: failed to write file '%s': %s", save_path, e)
    
    @ell.complex(model='gpt-4o', client=OpenAI(api_key='[KEEP OUT!]'), tools=[collision_predictor])
    def chat_with_tools(messages):
        """
        ell handles the conversion and API call automatically. 
        Just return the messages passed from the chat widget.
        """
        # Simply return the messages; ell expects a list of dicts or messages
        # marimo's mo.ui.chat provides objects that need conversion to simple dicts
        return [
            {"role": m.role if hasattr(m, 'role') else m.get('role', 'user'), 
            "content": m.content if hasattr(m, 'content') else m.get('content', '')}
            for m in messages
        ]

    def response_handler(messages):
        response = chat_with_tools(messages)
        
        if response.tool_calls:
            # Execute tool and get the raw probability
            prob = response.tool_calls[0]() 
            # Manually return a descriptive string
            return f"I've analyzed the video. The collision risk probability is {prob:.2%}. " \
                f"{'This indicates a high risk!' if prob > 0.75 else 'The risk appears low.'}"
            
        return response.text
    
    mo.ui.chat(
        response_handler,
        prompts=[
            "Analyze the video at C:\\path\\to\\video.mp4 for collision risk.",
            "What is the collision probability for the video located at C:\\videos\\test.mp4?",
        ],
    )

    return

# @app.cell
# def _eval_controls_ui():
#     run_jsonl_btn = mo.ui.button(label="Run JSONL Eval")
#     run_agent_btn = mo.ui.button(label="Run Agent Eval")
#     status = mo.ui.text()
#     output = mo.ui.text()
#     mo.vstack([mo.hstack([run_jsonl_btn, run_agent_btn]), status, output])
#     return run_jsonl_btn, run_agent_btn, status, output

# @app.cell
# def _eval_controls_logic(run_jsonl_btn, run_agent_btn, status, output):
#     import os as _os
#     import json as _json
#     from evals import (
#         run_mock_evaluation_from_jsonl,
#         rows_from_jsonl,
#         run_collision_evaluation_with_agent,
#         run_mock_evaluation,
#     )
#     # Explicit reactive dependency on button values to ensure re-execution on click
#     _ = (run_jsonl_btn.value, run_agent_btn.value)

#     def find_jsonl_path():
#         env_p = _os.getenv("COLLECT_PATH")
#         if env_p and _os.path.exists(env_p):
#             return env_p
#         cwd_p = _os.path.join(_os.getcwd(), "data", "collected.jsonl")
#         if _os.path.exists(cwd_p):
#             return cwd_p
#         this_dir = _os.path.dirname(__file__)
#         ws_p = _os.path.join(_os.path.dirname(this_dir), "data", "collected.jsonl")
#         if _os.path.exists(ws_p):
#             return ws_p
#         return None

#     project = _os.getenv("WEAVE_PROJECT", "WeaveHacks3-Evals")
#     threshold = float(_os.getenv("THRESHOLD", "0.75"))

#     if run_jsonl_btn.value:
#         status.value = "Running JSONL evaluation…"
#         jp = find_jsonl_path()
#         try:
#             if jp:
#                 res = run_mock_evaluation_from_jsonl(jp, threshold=threshold, project=project)
#             else:
#                 rows = [
#                     {"id": "0", "video_path": "00005.mp4", "target": {"label": True}, "mock_prob": 0.92},
#                     {"id": "1", "video_path": "01234.mp4", "target": {"label": False}, "mock_prob": 0.18},
#                     {"id": "2", "video_path": "00036.mp4", "target": {"label": True}, "mock_prob": 0.55},
#                 ]
#                 res = run_mock_evaluation(rows, threshold=threshold, project=project)
#             output.value = _json.dumps(res, indent=2)
#             status.value = "JSONL evaluation complete"
#         except Exception as e:
#             output.value = f"JSONL eval failed: {e}"
#             status.value = "JSONL evaluation failed"

#     if run_agent_btn.value:
#         status.value = "Running agent evaluation…"
#         jp = find_jsonl_path()
#         try:
#             if jp:
#                 rows = rows_from_jsonl(jp)
#                 for r in rows:
#                     r.pop("mock_prob", None)
#             else:
#                 rows = [
#                     {"id": "0", "video_path": "00005.mp4", "target": {"label": True}},
#                     {"id": "1", "video_path": "01234.mp4", "target": {"label": False}},
#                     {"id": "2", "video_path": "00036.mp4", "target": {"label": True}},
#                 ]
#             res = run_collision_evaluation_with_agent(rows, threshold=threshold, project=project)
#             output.value = _json.dumps(res, indent=2)
#             status.value = "Agent evaluation complete"
#         except Exception as e:
#             output.value = f"Agent eval failed: {e}"
#             status.value = "Agent evaluation failed"
#     return (run_jsonl_btn, run_agent_btn, status, output)

if __name__ == "__main__":
    app.run()
