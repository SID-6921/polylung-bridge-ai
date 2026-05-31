# Model Card: PolyLung Bridge AI (Mock Integration Build)

## Summary
This repository provides an end-to-end prototype connecting polymer classification outputs to lung inflammation risk scoring.

## Components
- Module 1: Polymer inference API (mocked class output in this build)
- Module 2 bridge: PSPII calculation + Bridge Score connector
- Streamlit dashboard: interactive scoring and JSON output display

## Intended Use
- Early feasibility and integration testing
- API contract verification with partner modules

## Limitations
- Current classifier output is mocked for integration speed
- Real image-based model inference hooks are scaffolded but not trained in this repo
- Not for clinical decision-making

## Metrics Tracked
- API latency
- Response success rate
- Bridge Score consistency across exposure and vulnerability gradients
