import argparse
import json
import os
import time
import pathlib
import logging


def _dec(v):
    if isinstance(v, bytes):
        try:
            return v.decode("utf-8", errors="replace")
        except Exception:
            return v
    return v


def build_default_bins(bin_count: int):
    step = 1.0 / max(2, bin_count)
    bins = []
    for i in range(bin_count):
        low = i * step
        high = (i + 1) * step if i < bin_count - 1 else 1.0
        bins.append({"low": low, "high": high, "rate": None})
    return bins


def write_calibration(save_path: str, bins, threshold: float, source: str):
    out = {
        "bins": bins,
        "threshold": float(threshold),
        "updated_ts": time.time(),
        "source": source,
    }
    pathlib.Path(save_path).parent.mkdir(parents=True, exist_ok=True)
    with open(save_path, "w", encoding="utf-8") as fh:
        fh.write(json.dumps(out, indent=2))
    logging.info("Wrote calibration file '%s'", save_path)


def calibrate_from_redis(
    redis_url: str | None,
    host: str,
    port: int,
    db: int,
    stream_key: str,
    out_path: str,
    bin_count: int,
    max_read: int,
    min_records: int,
    bootstrap: bool,
):
    try:
        import redis as _redis
    except Exception:
        if bootstrap:
            bins = build_default_bins(bin_count)
            write_calibration(out_path, bins, 0.75, "bootstrap_no_redis")
            return 0
        raise RuntimeError("redis package not available; install redis or use --bootstrap")

    r = _redis.from_url(redis_url) if redis_url else _redis.Redis(host=host, port=port, db=db)

    try:
        entries = r.xrevrange(stream_key, "+", "-", count=max_read)
    except Exception as e:
        if bootstrap:
            bins = build_default_bins(bin_count)
            write_calibration(out_path, bins, 0.75, "bootstrap_stream_error")
            return 0
        raise RuntimeError(f"Failed to read stream '{stream_key}': {e}")

    probs: list[float] = []
    labels: list[bool | None] = []
    decisions: list[bool | None] = []

    for _, fields in entries:
        j = _dec(fields.get("json"))
        if not j:
            continue
        try:
            ev = json.loads(j)
        except Exception:
            continue
        try:
            pr = float(((ev.get("prediction") or {}).get("probability")))
        except Exception:
            continue
        probs.append(pr)
        # decision
        try:
            decisions.append(bool(((ev.get("prediction") or {}).get("decision"))))
        except Exception:
            decisions.append(None)
        # label
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

    if len(probs) < min_records and not bootstrap:
        raise RuntimeError(f"Insufficient records ({len(probs)} < {min_records}); rerun after emitting more evals or pass --bootstrap")

    # Build equal-width bins
    step = 1.0 / max(2, bin_count)
    bins = []
    for i in range(bin_count):
        low = i * step
        high = (i + 1) * step if i < bin_count - 1 else 1.0
        idxs = [k for k, p in enumerate(probs) if low <= p <= high]
        vals = [labels[k] for k in idxs if labels[k] is not None]
        if len(vals) >= 3:
            rate = sum(1 for v in vals if v) / float(len(vals))
        else:
            dvals = [decisions[k] for k in idxs if decisions[k] is not None]
            rate = (sum(1 for v in dvals if v) / float(len(dvals))) if dvals else None
        bins.append({"low": low, "high": high, "rate": rate})

    # Threshold: optimize F1 if labels exist, else keep existing or 0.75
    chosen_threshold = None
    if any(v is not None for v in labels):
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

    # Merge existing threshold if no labels
    existing_threshold = 0.75
    try:
        if os.path.exists(out_path):
            with open(out_path, "r", encoding="utf-8") as fh:
                ex = json.load(fh)
            if isinstance(ex, dict) and "threshold" in ex:
                existing_threshold = float(ex["threshold"])  # preserve prior
    except Exception:
        pass

    final_threshold = float(chosen_threshold if chosen_threshold is not None else existing_threshold)
    write_calibration(out_path, bins, final_threshold, "redis_evals_cli")
    return 0


def main():
    ap = argparse.ArgumentParser(description="Generate collision_calibration.json from Redis evals stream.")
    ap.add_argument("--out", default=os.path.join(os.path.dirname(__file__), "collision_calibration.json"), help="Output calibration file path")
    ap.add_argument("--redis-url", default=os.getenv("REDIS_EVALS_URL"), help="Redis URL, e.g., redis://localhost:6379/0")
    ap.add_argument("--host", default=os.getenv("REDIS_HOST", "localhost"))
    ap.add_argument("--port", type=int, default=int(os.getenv("REDIS_PORT", "6379")))
    ap.add_argument("--db", type=int, default=int(os.getenv("REDIS_DB", "0")))
    ap.add_argument("--stream-key", default=os.getenv("REDIS_EVALS_STREAM_KEY", "evals:events"))
    ap.add_argument("--bin-count", type=int, default=int(os.getenv("CALIBRATION_BIN_COUNT", "10")))
    ap.add_argument("--max-read", type=int, default=int(os.getenv("CALIBRATION_MAX_READ", "1000")))
    ap.add_argument("--min-records", type=int, default=int(os.getenv("CALIBRATION_MIN_RECORDS", "50")))
    ap.add_argument("--bootstrap", action="store_true", help="Write default calibration if Redis missing or stream empty")
    ap.add_argument("--log-level", default=os.getenv("LOG_LEVEL", "INFO"))
    args = ap.parse_args()

    logging.basicConfig(level=getattr(logging, args.log_level.upper(), logging.INFO), format="[%(levelname)s] %(message)s")

    try:
        rc = calibrate_from_redis(
            redis_url=args.redis_url,
            host=args.host,
            port=args.port,
            db=args.db,
            stream_key=args.stream_key,
            out_path=args.out,
            bin_count=args.bin_count,
            max_read=args.max_read,
            min_records=args.min_records,
            bootstrap=args.bootstrap,
        )
    except Exception as e:
        logging.error("Calibration failed: %s", e)
        return 1
    return rc


if __name__ == "__main__":
    raise SystemExit(main())
