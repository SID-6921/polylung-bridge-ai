import argparse
import statistics
import time

import requests


def main() -> None:
    parser = argparse.ArgumentParser(description="Run 100-call integration feasibility benchmark.")
    parser.add_argument("--url", default="http://localhost:8000/analyze")
    parser.add_argument("--calls", type=int, default=100)
    args = parser.parse_args()

    latencies = []
    success = 0

    payload = {
        "exposure_route": "ingestion",
        "income_index": 1.0,
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

    print(f"calls={args.calls}")
    print(f"success_rate={success_rate:.3f}")
    print(f"latency_ms_p50={p50:.2f}")
    print(f"latency_ms_p95={p95:.2f}")


if __name__ == "__main__":
    main()
