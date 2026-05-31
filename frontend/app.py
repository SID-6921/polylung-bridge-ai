import os

import requests
import streamlit as st

API_URL = os.getenv("API_URL", "http://localhost:8000")

st.set_page_config(page_title="PolyLung Bridge AI", layout="wide")
st.title("PolyLung Bridge AI Dashboard")
st.caption("Module 1 + mock Module 2 bridge for polymer-resolved lung risk")

uploaded = st.file_uploader("Upload microscopy image", type=["png", "jpg", "jpeg", "tif", "tiff"])
exposure = st.selectbox("Exposure route", ["ingestion", "inhalation", "dermal"], index=0)
income_index = st.slider("Community vulnerability index", min_value=0.5, max_value=1.5, value=1.0, step=0.1)

if st.button("Analyze"):
    payload = {
        "exposure_route": exposure,
        "income_index": income_index,
    }

    try:
        response = requests.post(f"{API_URL}/analyze", json=payload, timeout=10)
        response.raise_for_status()
        data = response.json()

        c1, c2, c3 = st.columns(3)
        c1.metric("Polymer", data["polymer_type"])
        c2.metric("Bridge Score", data["bridge_score"])
        c3.metric("Risk Tier", data["risk_tier"])

        st.subheader("Raw Output")
        st.json(data)
    except Exception as exc:
        st.error(f"API call failed: {exc}")

if uploaded is not None:
    st.info("Image received. Current mock flow uses metadata only; image model hook is ready for Phase 2.")
