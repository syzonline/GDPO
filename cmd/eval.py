#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import argparse
import json
import math
import sys
from collections import defaultdict

def to_float(x):
    if x is None:
        return None
    if isinstance(x, (int, float)):
        if isinstance(x, float) and (math.isnan(x) or math.isinf(x)):
            return None
        return float(x)
    if isinstance(x, str):
        try:
            v = float(x.strip())
            if math.isnan(v) or math.isinf(v):
                return None
            return v
        except Exception:
            return None
    return None

def iter_json_objects(fp, debug=False, max_debug_chars=300):
    """
    Stream parser for:
      - JSON Lines (one object per line)
      - Multi-line JSON objects concatenated with whitespace/newlines
    Uses JSONDecoder.raw_decode to extract one complete object at a time.
    """
    decoder = json.JSONDecoder()
    buf = ""
    total_read = 0

    while True:
        chunk = fp.read(1024 * 1024)  # 1MB chunks
        if not chunk:
            break
        total_read += len(chunk)
        buf += chunk

        while True:
            s = buf.lstrip()
            if not s:
                buf = ""
                break
            offset = len(buf) - len(s)

            try:
                obj, idx = decoder.raw_decode(s)
            except json.JSONDecodeError as e:
                # Not enough data yet OR invalid JSON
                # Keep reading more; if debug, show context occasionally.
                if debug:
                    ctx_start = max(0, offset + e.pos - 80)
                    ctx_end = min(len(buf), offset + e.pos + 80)
                    ctx = buf[ctx_start:ctx_end]
                    print(f"[debug] JSONDecodeError: {e.msg} at pos={offset+e.pos}, "
                          f"line={e.lineno}, col={e.colno}", file=sys.stderr)
                    print(f"[debug] context: {ctx!r}", file=sys.stderr)
                break

            consumed = offset + idx
            yield obj
            buf = buf[consumed:]

    # Try parse leftover buffer
    s = buf.strip()
    if s:
        try:
            obj = json.loads(s)
            yield obj
        except json.JSONDecodeError as e:
            # This usually indicates truly invalid JSON (e.g., unescaped newlines/control chars).
            ctx = s[:max_debug_chars]
            raise RuntimeError(
                "文件尾部仍残留无法解析的内容：很可能存在未转义换行/控制字符，或日志并非合法 JSON。\n"
                f"json error: {e.msg} (line={e.lineno}, col={e.colno})\n"
                f"leading leftover: {ctx!r}"
            )

def update_stat(stat, v):
    stat["n"] += 1
    stat["sum"] += v
    stat["sumsq"] += v * v
    stat["min"] = v if stat["min"] is None else min(stat["min"], v)
    stat["max"] = v if stat["max"] is None else max(stat["max"], v)

def finalize(stat):
    n = stat["n"]
    if n == 0:
        return 0, float("nan"), float("nan"), None, None
    mean = stat["sum"] / n
    var = stat["sumsq"] / n - mean * mean
    std = math.sqrt(var) if var > 0 else 0.0
    return n, mean, std, stat["min"], stat["max"]

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("path", help="Path to *.jsonl (or multi-line JSON objects file)")
    ap.add_argument("--sep", default="\t", help="Output separator (default: TAB)")
    ap.add_argument("--wide", action="store_true", help="Wide output (one row per data_source, mean of each score key)")
    ap.add_argument("--debug", action="store_true", help="Print JSON decode debug context")
    args = ap.parse_args()

    stats = defaultdict(lambda: defaultdict(lambda: {"n": 0, "sum": 0.0, "sumsq": 0.0, "min": None, "max": None}))
    ds_count = defaultdict(int)
    all_keys = set()

    # utf-8-sig handles BOM; errors='replace' avoids hard crash on bad bytes
    with open(args.path, "r", encoding="utf-8-sig", errors="replace") as fp:
        for rec in iter_json_objects(fp, debug=args.debug):
            ds = rec.get("data_source", "UNKNOWN")
            ds_count[ds] += 1
            scores = rec.get("scores", {})
            if not isinstance(scores, dict):
                continue
            for k, raw in scores.items():
                v = to_float(raw)
                if v is None:
                    continue
                all_keys.add(k)
                update_stat(stats[ds][k], v)

    sep = args.sep
    dss = sorted(ds_count.keys())
    keys = sorted(all_keys)

    if not args.wide:
        print(sep.join(["data_source", "score_key", "n", "mean", "std", "min", "max"]))
        for ds in dss:
            for k in keys:
                n, mean, std, mn, mx = finalize(stats[ds][k])
                if n == 0:
                    continue
                print(sep.join([
                    ds, k, str(n),
                    f"{mean:.6g}", f"{std:.6g}",
                    f"{mn:.6g}" if mn is not None else "NA",
                    f"{mx:.6g}" if mx is not None else "NA",
                ]))
    else:
        header = ["data_source", "n_records"] + [f"{k}_mean" for k in keys]
        print(sep.join(header))
        for ds in dss:
            row = [ds, str(ds_count[ds])]
            for k in keys:
                n, mean, *_ = finalize(stats[ds][k])
                row.append(f"{mean:.6g}" if n else "NA")
            print(sep.join(row))

if __name__ == "__main__":
    main()
