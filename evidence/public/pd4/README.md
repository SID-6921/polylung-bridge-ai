# PD-4 API Integration Benchmark

This benchmark runs 100 calls to the mock PSPII endpoint and stores a structured report.

Scope clarification:
- This is an integration-contract and API-overhead benchmark.
- It does not represent full image-model inference latency.

Run:
```bash
python scripts/integration_benchmark.py --url http://localhost:8000/pspii --calls 100 --polymer PS --output evidence/public/pd4/benchmark_result.json
```

Output fields:
- `success_rate`
- `latency_ms_p50`
- `latency_ms_p95`
- `calls`
- `polymer`
