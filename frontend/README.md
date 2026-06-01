# PolyLung Bridge-AI Module

This repository contains the refactored backend validation, configuration paths, and secrets management updates for the PolyLung Bridge-AI module.

## 🛠️ Completed Deliverables
1. **User Control Inputs:** Streamlit UI parameters pass dynamically to backend computations.
2. **Pipeline Limitations Disclosure:** Added a prominent `st.warning` banner inside `frontend/app.py`.
3. **Secret Security Management:** Erased hardcoded Census API keys; system references server environment variables.
4. **Startup I/O Configuration:** Refactored PSPII file checker inside `scoring.py` to validate paths and cache data configurations at boot time.
5. **Request Safety Filters:** Built complete error exception rules checking ZIP layouts, bounds, and Census timeout caps.
6. **Testing Architecture:** Added `test_scoring.py` with automated integration assertions (3/3 tests passing).
7. **Packaging Cleanup:** Restructured project layers and added a standard root-level `Dockerfile`.

---

## 🚀 Local Startup & Execution Commands

To run and verify this branch immediately from the project root directory, use the following execution sequences:

### 1. Run the Automated Test Engine Suite
```bash
python3 -m pytest test_scoring.py