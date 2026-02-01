import os
import time
import json
import argparse
from typing import Optional

try:
    import redis
except Exception:
    redis = None


def get_redis_client() -> "redis.Redis":
    if redis is None:
        raise RuntimeError("redis library not installed. Run: pip install redis")
    url = os.getenv("REDIS_URL")
    if url:
        return redis.from_url(url)
    host = os.getenv("REDIS_HOST", "localhost")
    port = int(os.getenv("REDIS_PORT", "6379"))
    db = int(os.getenv("REDIS_DB", "0"))
    return redis.Redis(host=host, port=port, db=db)


def pretty_print(record: dict, pretty: bool = True) -> None:
    if pretty:
        print(json.dumps(record, indent=2, ensure_ascii=False))
    else:
        print(json.dumps(record, ensure_ascii=False))


def decode_stream_entry(entry) -> dict:
    """Decode a Redis Stream entry to a dict.
    entry: tuple like (id, {field: value, ...})
    """
    _id, fields = entry
    # Decode entry id if bytes
    try:
        if isinstance(_id, bytes):
            _id = _id.decode("utf-8", errors="replace")
    except Exception:
        pass
    out = {"_id": _id}
    # Decode keys and values to strings when needed
    for k, v in fields.items():
        try:
            if isinstance(k, bytes):
                k = k.decode("utf-8", errors="replace")
        except Exception:
            pass
        try:
            if isinstance(v, bytes):
                v = v.decode("utf-8", errors="replace")
        except Exception:
            pass
        out[k] = v
    # If it contains a JSON blob under 'json', parse it
    if "json" in out:
        try:
            out["json_parsed"] = json.loads(out["json"])  # keep original under 'json'
        except Exception:
            pass
    return out


def _normalize_value(val: str):
    """Convert filter value strings to bool/float/int when appropriate."""
    v = str(val).strip().lower()
    if v in ("true", "false"):
        return v == "true"
    try:
        if "." in v:
            return float(v)
        return int(v)
    except Exception:
        return val


def _get_field(rec: dict, key: str):
    """Try to get a field from record; check top-level and json_parsed."""
    if key in rec:
        return rec[key]
    jp = rec.get("json_parsed")
    if isinstance(jp, dict) and key in jp:
        return jp[key]
    return None


def _matches_filters(rec: dict, filters: list[tuple[str, object]]) -> bool:
    for k, v in filters:
        rv = _get_field(rec, k)
        if rv is None:
            return False
        # Normalize booleans/strings for comparison
        if isinstance(rv, str):
            rvs = rv.strip().lower()
            if rvs in ("true", "false"):
                rv = rvs == "true"
        if rv != v:
            return False
    return True


def tail_stream(
    r: "redis.Redis",
    key: str,
    start: str = "0-0",
    block_ms: int = 5000,
    pretty: bool = True,
    follow: bool = False,
    max_count: int = 0,
    idle_exit_ms: int = 0,
    filters: list[tuple[str, object]] | None = None,
) -> None:
    last_id = start
    consumed = 0
    last_activity = time.time()

    # Initial read (backlog)
    initial = r.xread({key: last_id}, count=100)
    for _, entries in initial:
        for e in entries:
            rec = decode_stream_entry(e)
            if not filters or _matches_filters(rec, filters):
                pretty_print(rec, pretty)
            last_id = e[0]
            consumed += 1
            last_activity = time.time()
            if max_count > 0 and consumed >= max_count:
                return

    # If not following, exit after backlog
    if not follow:
        return

    # Tail new entries
    while True:
        try:
            res = r.xread({key: last_id}, block=block_ms)
            if not res:
                if idle_exit_ms > 0 and (time.time() - last_activity) * 1000 >= idle_exit_ms:
                    return
                continue
            for _, entries in res:
                for e in entries:
                    rec = decode_stream_entry(e)
                    if not filters or _matches_filters(rec, filters):
                        pretty_print(rec, pretty)
                    last_id = e[0]
                    consumed += 1
                    last_activity = time.time()
                    if max_count > 0 and consumed >= max_count:
                        return
        except KeyboardInterrupt:
            print("\nStopped.")
            return
        except Exception as e:
            print(f"Stream read error: {e}")
            time.sleep(1)


def tail_list(
    r: "redis.Redis",
    key: str,
    timeout: int = 5,
    pretty: bool = True,
    follow: bool = True,
    max_count: int = 0,
    filters: list[tuple[str, object]] | None = None,
) -> None:
    consumed = 0
    while True:
        try:
            item = r.blpop(key, timeout=timeout)
            if item is None:
                # If not following (one-shot) and nothing was consumed, exit
                if not follow:
                    return
                continue  # timeout, loop again
            _, raw = item
            try:
                if isinstance(raw, bytes):
                    raw = raw.decode("utf-8", errors="replace")
            except Exception:
                pass
            try:
                obj = json.loads(raw)
            except Exception:
                obj = {"raw": raw}
            if not filters or _matches_filters(obj, filters):
                pretty_print(obj, pretty)
            consumed += 1
            if max_count > 0 and consumed >= max_count:
                return
            if not follow:
                return
        except KeyboardInterrupt:
            print("\nStopped.")
            return
        except Exception as e:
            print(f"List read error: {e}")
            time.sleep(1)


def main():
    parser = argparse.ArgumentParser(description="Redis consumer for collision events")
    parser.add_argument("--mode", choices=["stream", "list"], default="stream", help="Consume from Redis stream or list")
    parser.add_argument("--key", default=None, help="Stream/List key to consume")
    parser.add_argument("--start", default="0-0", help="Stream start id (use '$' to start from latest)")
    parser.add_argument("--block-ms", type=int, default=5000, help="Block time for stream reads in ms")
    parser.add_argument("--list-timeout", type=int, default=5, help="BLPOP timeout in seconds")
    parser.add_argument("--compact", action="store_true", help="Output compact JSON (no pretty printing)")
    parser.add_argument("--follow", action="store_true", help="Keep tailing; otherwise exit after initial read (stream) or first item (list)")
    parser.add_argument("--max", type=int, default=0, help="Max entries to print before exiting (0 = unlimited)")
    parser.add_argument("--idle-exit-ms", type=int, default=0, help="Exit if idle for this many milliseconds (stream mode)")
    parser.add_argument("--filter", action="append", default=[], help="Filter as field=value (repeatable)")
    parser.add_argument("--task", default=None, help="Shortcut: filter by evaluation 'task'")
    parser.add_argument("--decision", default=None, help="Shortcut: filter by decision=true|false")
    args = parser.parse_args()

    r = get_redis_client()

    # Build filters
    filt: list[tuple[str, object]] = []
    for f in args.filter:
        if "=" in f:
            k, v = f.split("=", 1)
            filt.append((k.strip(), _normalize_value(v)))
    if args.task:
        filt.append(("task", args.task))
    if args.decision is not None:
        filt.append(("decision", _normalize_value(args.decision)))

    if args.mode == "stream":
        key = args.key or os.getenv("REDIS_STREAM_KEY", "collision:events")
        print(f"Consuming stream '{key}' from start '{args.start}' (block {args.block_ms} ms, follow={args.follow})...")
        tail_stream(
            r,
            key,
            start=args.start,
            block_ms=args.block_ms,
            pretty=(not args.compact),
            follow=args.follow,
            max_count=args.max,
            idle_exit_ms=args.idle_exit_ms,
            filters=filt or None,
        )
    else:
        key = args.key or os.getenv("REDIS_LIST_KEY", "collision:events-list")
        print(f"Consuming list '{key}' (timeout {args.list_timeout}s, follow={args.follow})...")
        tail_list(
            r,
            key,
            timeout=args.list_timeout,
            pretty=(not args.compact),
            follow=args.follow,
            max_count=args.max,
            filters=filt or None,
        )


if __name__ == "__main__":
    main()
