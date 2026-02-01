import os
import sys
import json
from typing import Optional

from evals import (
    run_mock_evaluation,
    rows_from_jsonl,
    run_collision_evaluation_with_agent,
    build_evaluation,
    compute_metrics_for_prediction,
    send_evaluation,
)

try:
    import weave
except Exception:
    weave = None


def find_jsonl_path(env_var: str = "COLLECT_PATH") -> Optional[str]:
    # Prefer explicit env var
    p = os.getenv(env_var)
    if p and os.path.exists(p):
        return p

    # Try CWD/data/collected.jsonl
    cwd_candidate = os.path.join(os.getcwd(), "data", "collected.jsonl")
    if os.path.exists(cwd_candidate):
        return cwd_candidate

    # Try workspace-root/data/collected.jsonl (parent of WeaveHacks3)
    this_dir = os.path.dirname(__file__)
    parent = os.path.dirname(this_dir)
    ws_candidate = os.path.join(parent, "data", "collected.jsonl")
    if os.path.exists(ws_candidate):
        return ws_candidate

    return None


def sample_rows_for_mock():
    # Minimal harness rows (no video required), includes mock_prob
    return [
        {"id": "0", "video_path": "00005.mp4", "target": {"label": True}, "mock_prob": 0.92},
        {"id": "1", "video_path": "01234.mp4", "target": {"label": False}, "mock_prob": 0.18},
        {"id": "2", "video_path": "00036.mp4", "target": {"label": True}, "mock_prob": 0.55},
    ]


def sample_rows_for_agent():
    # Agent-backed eval does not use mock_prob; only video_path and target
    return [
        {"id": "0", "video_path": "00005.mp4", "target": {"label": True}},
        {"id": "1", "video_path": "01234.mp4", "target": {"label": False}},
        {"id": "2", "video_path": "00036.mp4", "target": {"label": True}},
    ]


if __name__ == "__main__":
    project = os.getenv("WEAVE_PROJECT", "WeaveHacks3-Evals")
    threshold = float(os.getenv("THRESHOLD", "0.75"))
    mode = os.getenv("EVAL_MODE", "jsonl").lower()
    model_version = os.getenv("MODEL_VERSION", "v1.0")
    session_id = os.getenv("SESSION_ID")
    ingest_url = os.getenv("WEAVE_INGEST_URL")
    ingest_headers = None
    emit_only = os.getenv("EMIT_ONLY", "0") == "1"  # if set, skip weave evaluation and only emit
    client = None
    try:
        if weave is not None:
            client = weave.init(project)
    except Exception:
        client = None

    # CLI override: python run_eval.py jsonl|agent
    if len(sys.argv) > 1:
        mode = sys.argv[1].lower()

    print(f"Running evaluation in project: {project} (mode={mode}, threshold={threshold})")

    if mode == "jsonl":
        jsonl_path = find_jsonl_path()
        if jsonl_path:
            print(f"Using JSONL: {jsonl_path}")
            rows = rows_from_jsonl(jsonl_path)
        else:
            print("No JSONL found; falling back to sample mock rows.")
            rows = sample_rows_for_mock()

        # Emit evaluations per-row (mirrors to Redis if enabled)
        for r in rows:
            prob = float(r.get("mock_prob", 0.5))
            decision = bool(prob > threshold)
            eval_payload = build_evaluation(
                model_version=model_version,
                task="collision_detection",
                input_obj={"video_path": r.get("video_path")},
                prediction=prob,
                decision=decision,
                session_id=session_id,
                metadata={"source": "run_eval.mock", "threshold": threshold},
            )
            # Attach label if available
            tgt = r.get("target") or {}
            if "label" in tgt and tgt["label"] is not None:
                try:
                    eval_payload["reference"] = {"label": int(bool(tgt["label"]))}
                except Exception:
                    eval_payload["reference"] = {"label": None}
            eval_payload["metrics"] = compute_metrics_for_prediction(prob, threshold=threshold)
            send_evaluation(client=client, evaluation=eval_payload, ingest_url=ingest_url, ingest_headers=ingest_headers)

        result = None if emit_only else run_mock_evaluation(rows, threshold=threshold, project=project)
    elif mode == "agent":
        jsonl_path = find_jsonl_path()
        if jsonl_path:
            print(f"Using JSONL to derive rows: {jsonl_path}")
            rows = rows_from_jsonl(jsonl_path)
            for r in rows:
                r.pop("mock_prob", None)
        else:
            print("No JSONL found; falling back to sample agent rows.")
            rows = sample_rows_for_agent()

        # Emit evaluations by running the agent locally (lightweight mirror)
        try:
            from marimofiguration import CollisionVideoModel, CollisionDetectionAgent
            import os as _os
            model = CollisionVideoModel()
            agent = CollisionDetectionAgent(model, threshold=threshold)
            for r in rows:
                vp = r.get("video_path", "")
                if not _os.path.exists(vp):
                    prob = 0.5
                    decision = False
                else:
                    t = agent.perceive(vp)
                    prob = float(agent.think(t))
                    # use agent's decision method if present; else threshold
                    try:
                        decision = bool(agent.decide(prob))
                    except Exception:
                        decision = bool(prob > threshold)

                eval_payload = build_evaluation(
                    model_version=model_version,
                    task="collision_detection",
                    input_obj={"video_path": vp},
                    prediction=prob,
                    decision=decision,
                    session_id=session_id,
                    metadata={"source": "run_eval.agent", "threshold": threshold},
                )
                tgt = r.get("target") or {}
                if "label" in tgt and tgt["label"] is not None:
                    try:
                        eval_payload["reference"] = {"label": int(bool(tgt["label"]))}
                    except Exception:
                        eval_payload["reference"] = {"label": None}
                eval_payload["metrics"] = compute_metrics_for_prediction(prob, threshold=threshold)
                send_evaluation(client=client, evaluation=eval_payload, ingest_url=ingest_url, ingest_headers=ingest_headers)
        except Exception as e:
            print(f"Agent emit failed: {e}")

        result = None if emit_only else run_collision_evaluation_with_agent(rows, threshold=threshold, project=project)
    else:
        print("Unknown mode. Use 'jsonl' or 'agent'.")
        sys.exit(1)

    if result is not None:
        print("Evaluation finished. Result:")
        print(result)
    else:
        print("Emit-only mode: evaluations sent; no weave evaluation run.")
