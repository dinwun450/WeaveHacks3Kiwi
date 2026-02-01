import time
import uuid
import os
import json

# Optional Weave imports for full evaluation pipeline
try:
    import asyncio
    import weave
    from weave.scorers import MultiTaskBinaryClassificationF1
except Exception:
    weave = None
    asyncio = None
    MultiTaskBinaryClassificationF1 = None

try:
    import requests
except Exception:
    requests = None

# Optional Redis will be initialized lazily inside functions


def _init_redis_for_evals():
    """Return a redis client if evals logging is enabled via env vars.

    Env vars:
      - REDIS_EVALS_ENABLE=1 to enable
      - REDIS_EVALS_URL or REDIS_EVALS_HOST/REDIS_EVALS_PORT/REDIS_EVALS_DB
      - REDIS_EVALS_STREAM_KEY (default 'evals:events')
      - REDIS_EVALS_STREAM_MAXLEN (default '10000')
      - REDIS_EVALS_LIST_KEY (optional)
      - REDIS_EVALS_LIST_MAXLEN (optional)
    """
    try:
        if os.getenv("REDIS_EVALS_ENABLE", "0") != "1":
            return None
        import redis as _redis
        url = os.getenv("REDIS_EVALS_URL")
        if url:
            return _redis.from_url(url)
        host = os.getenv("REDIS_EVALS_HOST", os.getenv("REDIS_HOST", "localhost"))
        port = int(os.getenv("REDIS_EVALS_PORT", os.getenv("REDIS_PORT", "6379")))
        db = int(os.getenv("REDIS_EVALS_DB", os.getenv("REDIS_DB", "0")))
        return _redis.Redis(host=host, port=port, db=db)
    except Exception:
        return None


def log_evaluation_to_redis(evaluation: dict) -> bool:
    """Append evaluation payload to a Redis Stream or List if enabled.

    Returns True if a write was attempted and did not throw; False otherwise.
    """
    try:
        r = _init_redis_for_evals()
        if r is None:
            return False
        stream_key = os.getenv("REDIS_EVALS_STREAM_KEY", "evals:events")
        list_key = os.getenv("REDIS_EVALS_LIST_KEY")
        stream_maxlen = int(os.getenv("REDIS_EVALS_STREAM_MAXLEN", "10000"))
        list_maxlen = os.getenv("REDIS_EVALS_LIST_MAXLEN")

        payload = json.dumps(evaluation, default=str)
        # Prefer stream; fallback to list if configured explicitly
        if stream_key:
            fields = {
                "json": payload,
                "id": str(evaluation.get("id", "")),
                "task": str(evaluation.get("task", "")),
                "model_version": str(evaluation.get("model_version", "")),
            }
            r.xadd(stream_key, fields, maxlen=stream_maxlen, approximate=True)
            return True
        if list_key:
            r.lpush(list_key, payload)
            if list_maxlen is not None:
                try:
                    r.ltrim(list_key, 0, int(list_maxlen) - 1)
                except Exception:
                    pass
            return True
    except Exception:
        # do not propagate logging failures
        return False
    return False


def build_evaluation(model_version, task, input_obj, prediction, decision, session_id=None, metadata=None):
    """Return a standardized evaluation dict suitable for Weave Evaluations ingestion.

    Fields chosen to match the Weave Evaluations guide: id, ts, model_version,
    task, input, prediction, reference, metrics, provenance, artifacts.
    """
    eval_id = f"eval-{session_id or 'nosess'}-{int(time.time()*1000)}-{uuid.uuid4().hex[:8]}"
    return {
        "id": eval_id,
        "ts": time.time(),
        "model_version": model_version,
        "task": task,
        "input": input_obj,
        "prediction": {
            "probability": float(prediction),
            "decision": decision,
        },
        "reference": None,
        "metrics": {},
        "provenance": {"agent": "marimo", "session_id": session_id},
        "artifacts": {},
    }


def compute_metrics_for_prediction(prob, threshold=0.75):
    """Simple derived metrics for a single prediction."""
    return {
        "threshold": float(threshold),
        "is_high_risk": bool(float(prob) > float(threshold)),
        "probability": float(prob),
    }


def send_evaluation(client=None, evaluation=None, ingest_url=None, ingest_headers=None):
    """Send evaluation to Weave via SDK if available, otherwise to an ingest URL,
    otherwise fallback to printing the payload.

    - `client`: the `weave` client object (if it exposes `evaluations.create`).
    - `ingest_url` + `ingest_headers`: fallback HTTP ingest.
    """
    if evaluation is None:
        return False

    # Try SDK call first
    try:
        if client is not None and hasattr(client, "evaluations"):
            try:
                client.evaluations.create(evaluation)
                # Mirror to Redis if enabled
                log_evaluation_to_redis(evaluation)
                return True
            except Exception:
                # continue to HTTP fallback
                pass
    except Exception:
        pass

    # HTTP fallback
    if ingest_url and requests is not None:
        try:
            r = requests.post(ingest_url, headers=ingest_headers or {}, json=evaluation, timeout=5)
            r.raise_for_status()
            # Mirror to Redis if enabled
            log_evaluation_to_redis(evaluation)
            return True
        except Exception:
            pass

    # Final fallback: print (or log) the evaluation
    # Final fallback: print (or log) the evaluation and attempt Redis mirror
    try:
        print("[evals.send_evaluation]", json.dumps(evaluation, default=str))
    except Exception:
        print("[evals.send_evaluation] - could not serialize evaluation", evaluation)
    # Try Redis even on fallback
    log_evaluation_to_redis(evaluation)
    return False


def attach_reference(client=None, eval_id=None, label=None, annotator=None, notes=None, ingest_url=None, ingest_headers=None):
    """Attach a human reference label to an existing evaluation. Attempts SDK update,
    then HTTP fallback. If neither is available it prints the update payload.
    """
    if eval_id is None:
        raise ValueError("eval_id required")

    update = {
        "id": eval_id,
        "reference": {
            "label": int(label) if label is not None else None,
            "annotator": annotator,
            "notes": notes,
            "ts": time.time(),
        },
    }

    # Try SDK update
    try:
        if client is not None and hasattr(client, "evaluations") and hasattr(client.evaluations, "update"):
            try:
                client.evaluations.update(update)
                return True
            except Exception:
                pass
    except Exception:
        pass

    # HTTP fallback
    if ingest_url and requests is not None:
        try:
            r = requests.post(ingest_url, headers=ingest_headers or {}, json=update, timeout=5)
            r.raise_for_status()
            return True
        except Exception:
            pass

    print("[evals.attach_reference]", update)
    return False


# -------------------------------
# Weave tutorial-style pipeline
# -------------------------------

def _require_weave():
    if weave is None:
        raise RuntimeError(
            "weave is not installed. Please install it with `pip install weave`."
        )
    return weave


def _init_weave(project: str | None = None):
    wv = _require_weave()
    # Use provided project or fallback to env/default
    project_name = project or os.getenv("WEAVE_PROJECT", "eval_pipeline_quickstart")
    return wv.init(project_name)


def create_collision_dataset(rows):
    """Create a Weave Dataset for collision evaluation.

    rows should be a list of dicts with keys:
      - id: unique string id
      - video_path: path to video to score
      - target: dict with {'label': bool} ground truth (optional but recommended)

    Example row: {'id': '0', 'video_path': 'C:/videos/a.mp4', 'target': {'label': True}}
    """
    wv = _require_weave()
    dataset = wv.Dataset(name="collision_dataset", rows=rows)
    wv.publish(dataset)
    return dataset


def dataset_from_jsonl(jsonl_path: str, label_field: str = "label"):
    """Build a Weave Dataset from a JSONL file.

    The JSONL file can come from online collection and may contain keys like
    'video_path', 'prediction', 'decision', etc. If a `label_field` is present,
    it will be used as target label.
    """
    wv = _require_weave()
    rows = []
    with open(jsonl_path, "r", encoding="utf-8") as fh:
        for idx, line in enumerate(fh):
            try:
                rec = json.loads(line)
            except Exception:
                continue
            vid = rec.get("video_path") or rec.get("video") or rec.get("path")
            tgt = None
            if label_field in rec:
                tgt = {"label": bool(rec[label_field])}
            row = {
                "id": str(rec.get("id", idx)),
                "video_path": vid,
                "target": tgt,
            }
            # Map recorded prediction to mock_prob for replayable evaluations
            if "prediction" in rec:
                try:
                    row["mock_prob"] = float(rec["prediction"])
                except Exception:
                    pass
            rows.append(row)
    dataset = wv.Dataset(name=f"collision_dataset_{uuid.uuid4().hex[:6]}", rows=rows)
    wv.publish(dataset)
    return dataset


def collision_threshold_score_factory(threshold: float = 0.75):
    """Create a @weave.op scorer comparing ground-truth label vs model decision."""
    wv = _require_weave()

    @wv.op()
    def score(target: dict, output: dict) -> dict:
        # target: {'label': bool}; output: {'probability': float, 'decision': bool}
        if target is None or "label" not in target:
            # No label available; return empty metrics to avoid breaking the eval
            return {"has_label": False}
        return {
            "has_label": True,
            "correct": bool(target["label"]) == bool(output.get("decision")),
            "probability": float(output.get("probability", 0.0)),
        }

    return score


class MockCollisionWeaveModel:
    """A minimal model compatible with Weave evaluations for dry runs.

    It expects the dataset row to include 'mock_prob' or falls back to 0.5.
    This avoids heavy video dependencies while enabling a working evaluation pipeline.
    """

    def __init__(self, threshold: float = 0.75):
        self.threshold = float(threshold)

    def _predict_impl(self, sentence_or_path: str, mock_prob: float | None = None):
        prob = float(mock_prob) if mock_prob is not None else 0.5
        return {"probability": prob, "decision": prob > self.threshold}

    def predict(self, row):
        # Not an async weave.op to keep requirements light for the harness
        # Row may be a dict with keys video_path and optional mock_prob
        return self._predict_impl(row.get("video_path", ""), row.get("mock_prob"))


def run_mock_evaluation(rows, threshold: float = 0.75, project: str | None = None):
    """Run a tutorial-style evaluation using a mock model and provided rows.

    rows: list of {'id', 'video_path', 'target': {'label': bool}, 'mock_prob': float}
    Returns the evaluation result object from weave.
    """
    wv = _require_weave()
    _init_weave(project)

    dataset = wv.Dataset(name=f"collision_mock_{uuid.uuid4().hex[:4]}", rows=rows)
    wv.publish(dataset)

    def _has_label(_rows):
        try:
            return any((r.get("target") or {}).get("label") is not None for r in _rows)
        except Exception:
            return False

    # Built-in multi-task F1 on a single field only when labels exist
    scorers = []
    if MultiTaskBinaryClassificationF1 is not None and _has_label(rows):
        scorers.append(MultiTaskBinaryClassificationF1(class_names=["label"]))
    scorers.append(collision_threshold_score_factory(threshold))

    evaluation = wv.Evaluation(
        name=f"collision_eval_{uuid.uuid4().hex[:4]}",
        dataset=dataset,
        scorers=scorers,
    )

    model = MockCollisionWeaveModel(threshold=threshold)

    # Op to produce predictions from dataset row fields
    @wv.op()
    async def predict_op(video_path: str, mock_prob: float | None = None):
        return model._predict_impl(video_path, mock_prob)

    if asyncio is None:
        raise RuntimeError("asyncio not available; cannot run evaluation")

    # Pass the op directly to Evaluation.evaluate per tutorial API
    result = asyncio.run(evaluation.evaluate(predict_op))
    return result


def run_collision_evaluation_with_agent(rows, threshold: float = 0.75, project: str | None = None):
    """Advanced: Run evaluation with the real CollisionDetectionAgent if available.

    rows: list of {'id', 'video_path', 'target': {'label': bool}}
    This will attempt to import your Collision model and may require video deps.
    """
    wv = _require_weave()
    _init_weave(project)

    dataset = wv.Dataset(name=f"collision_agent_{uuid.uuid4().hex[:4]}", rows=rows)
    wv.publish(dataset)

    def _has_label(_rows):
        try:
            return any((r.get("target") or {}).get("label") is not None for r in _rows)
        except Exception:
            return False

    scorers = []
    if MultiTaskBinaryClassificationF1 is not None and _has_label(rows):
        scorers.append(MultiTaskBinaryClassificationF1(class_names=["label"]))
    scorers.append(collision_threshold_score_factory(threshold))

    evaluation = wv.Evaluation(
        name=f"collision_agent_eval_{uuid.uuid4().hex[:4]}",
        dataset=dataset,
        scorers=scorers,
    )

    # Define an op that calls your agent directly
    @wv.op()
    async def agent_predict(video_path: str) -> dict:
        from marimofiguration import CollisionVideoModel, CollisionDetectionAgent
        import os as _os
        model = CollisionVideoModel()
        agent = CollisionDetectionAgent(model, threshold=threshold)
        if not _os.path.exists(video_path):
            # Return an abstain-like output to keep evaluation flowing
            return {"probability": 0.5, "decision": False}
        tensor = agent.perceive(video_path)
        prob = float(agent.think(tensor))
        decision = bool(agent.decide(prob))
        return {"probability": prob, "decision": decision}

    if asyncio is None:
        raise RuntimeError("asyncio not available; cannot run evaluation")

    result = asyncio.run(evaluation.evaluate(agent_predict))
    return result


def rows_from_jsonl(jsonl_path: str, label_field: str = "label"):
    """Return rows list (id, video_path, target, mock_prob?) suitable for run_mock_evaluation.

    This mirrors dataset_from_jsonl but returns the raw list instead of a Weave Dataset.
    """
    rows = []
    with open(jsonl_path, "r", encoding="utf-8") as fh:
        for idx, line in enumerate(fh):
            try:
                rec = json.loads(line)
            except Exception:
                continue
            vid = rec.get("video_path") or rec.get("video") or rec.get("path")
            tgt = None
            if label_field in rec:
                tgt = {"label": bool(rec[label_field])}
            row = {
                "id": str(rec.get("id", idx)),
                "video_path": vid,
                "target": tgt,
            }
            if "prediction" in rec:
                try:
                    row["mock_prob"] = float(rec["prediction"])
                except Exception:
                    pass
            rows.append(row)
    return rows


def run_mock_evaluation_from_jsonl(jsonl_path: str, threshold: float = 0.75, project: str | None = None, label_field: str = "label"):
    """Convenience: build rows from a JSONL (e.g., collected.jsonl) and run mock evaluation."""
    rows = rows_from_jsonl(jsonl_path, label_field=label_field)
    return run_mock_evaluation(rows, threshold=threshold, project=project)

