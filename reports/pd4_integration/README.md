# PD-4 API Integration Benchmark

This benchmark runs 100 calls to the mock PSPII endpoint and stores a structured report.

Run:
```bash
python scripts/integration_benchmark.py --url http://localhost:8000/pspii --calls 100 --polymer PS --output reports/pd4_integration/benchmark_result.json
```

Output fields:
- `success_rate`
- `latency_ms_p50`
- `latency_ms_p95`
- `calls`
- `polymer`
