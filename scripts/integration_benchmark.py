import argparse
import json
from pathlib import Path
import statistics
import time

import requests


def main() -> None:
    parser = argparse.ArgumentParser(description="Run 100-call integration feasibility benchmark.")
    parser.add_argument("--url", default="http://localhost:8000/pspii")
    parser.add_argument("--calls", type=int, default=100)
    parser.add_argument("--polymer", default="PS")
    parser.add_argument("--output", default="evidence/public/pd4/benchmark_result.json")
    args = parser.parse_args()

    latencies = []
    success = 0

    payload = {
        "polymer_type": args.polymer,
    }

    for _ in range(args.calls):
        start = time.perf_counter()
        try:
            r = requests.post(args.url, json=payload, timeout=10)
            elapsed_ms = (time.perf_counter() - start) * 1000
            latencies.append(elapsed_ms)
            if r.status_code == 200:
                success += 1
        except Exception:
            elapsed_ms = (time.perf_counter() - start) * 1000
            latencies.append(elapsed_ms)

    success_rate = success / args.calls if args.calls else 0
    p50 = statistics.median(latencies) if latencies else 0
    p95 = sorted(latencies)[int(0.95 * (len(latencies) - 1))] if latencies else 0

    result = {
        "url": args.url,
        "calls": args.calls,
        "polymer": args.polymer,
        "success": success,
        "success_rate": round(success_rate, 4),
        "latency_ms_p50": round(p50, 2),
        "latency_ms_p95": round(p95, 2),
    }
    out_path = Path(args.output)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(result, indent=2), encoding="utf-8")

    print(f"calls={args.calls}")
    print(f"success_rate={success_rate:.3f}")
    print(f"latency_ms_p50={p50:.2f}")
    print(f"latency_ms_p95={p95:.2f}")
    print(f"output={out_path}")


if __name__ == "__main__":
    main()
