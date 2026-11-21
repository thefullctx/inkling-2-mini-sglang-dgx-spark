#!/usr/bin/env python3
import glob, json
from pathlib import Path

res_dir = Path(__file__).resolve().parent / "results"
print(f"{'Label':24s} | {'Smoke':5s} | {'c1 Short':9s} | {'TTFT':6s} | {'15K Cold':8s} | {'15K Warm':8s} | {'15K Dec':8s} | {'c4 Agg':8s} | {'RAM':6s}")
print("-" * 108)
for p in sorted(res_dir.glob("*.json")):
    d = json.load(open(p))
    ok = "PASS" if all(v.get("ok") for v in d["smoke"].values()) else "FAIL"
    print(
        f"{d['label']:24s} | {ok:5s} | "
        f"{d['c1_short']['avg_decode_tps']:6.2f} t/s | "
        f"{d['c1_short']['avg_ttft_s']:5.2f}s | "
        f"{d['c1_long']['cold']['ttft_s']:7.2f}s | "
        f"{d['c1_long']['warm']['ttft_s']:7.2f}s | "
        f"{d['c1_long']['warm']['decode_tps']:6.2f} t/s | "
        f"{d['c4_concurrent']['aggregate_tps']:6.2f} t/s | "
        f"{d['memory_mb'].get('used_mb',0)/1024:5.1f}G"
    )
