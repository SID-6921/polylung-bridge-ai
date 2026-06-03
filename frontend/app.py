import os

import requests
import streamlit as st

API_URL = os.getenv("API_URL", "http://localhost:8000")

st.set_page_config(page_title="PolyLung Bridge AI", layout="wide")
st.title("PolyLung Bridge AI Dashboard")
st.caption("Module 1 + mock Module 2 bridge for polymer-resolved lung risk")

uploaded = st.file_uploader("Upload microscopy image", type=["png", "jpg", "jpeg", "tif", "tiff"])
st.caption("Upload a microscopy image of the polymer sample for analysis.")

polymer = st.selectbox("Polymer type", ["PVC", "PS", "PU", "PE", "PP", "PET", "Nylon", "Acrylic", "PC", "ABS"])
st.caption("Select the type of polymer found in the sample.")

exposure = st.selectbox("Exposure route", ["ingestion", "inhalation", "dermal"], index=0)
st.caption("How the polymer enters the body — ingestion (swallowed), inhalation (breathed in), or dermal (skin contact).")

particle_count = st.number_input("Particle count", min_value=0, value=120, step=1)
st.caption("Number of microplastic particles detected in the sample.")

zipcode = st.text_input("ZIP code", max_chars=5)
st.caption("Your 5-digit ZIP code — used to calculate community vulnerability based on median area income.")



if st.button("Analyze"):
    if not uploaded:
        st.error("Please upload a microscopy image before analyzing.")
        st.stop()

    payload = {
        "polyType": polymer,
        "particleCount": str(particle_count),
        "zipcode": zipcode,
        "exposRoute": exposure,
    }

    try:
        files = {"file": (uploaded.name, uploaded.getvalue(), uploaded.type)}
        response = requests.post(f"{API_URL}/analyze", data=payload, files=files, timeout=10)
        response.raise_for_status()
        data = response.json()

        c1, c2, c3 = st.columns(3)
        c1.metric("Polymer", data["polymer_type"])
        c2.metric("Bridge Score", data["bridge_score"])
        c3.metric("Risk Tier", data["risk_tier"])

        st.subheader("Score Interpretation")
        tier = data["risk_tier"]
        if tier == "Low":
            st.success("Low Risk (score < 15) — Minimal concern. Standard monitoring recommended.")
        elif tier == "Elevated":
            st.warning("Elevated Risk (score 15-35) — Moderate concern. Increased monitoring advised.")
        elif tier == "High":
            st.error("High Risk (score 35-65) — Significant concern. Immediate review recommended.")
        elif tier == "Critical":
            st.error("Critical Risk (score > 65) — Severe concern. Urgent action required.")

        st.subheader("Raw Output")
        st.json(data)

        import json
        st.download_button(
            label="Download Results",
            data=json.dumps(data, indent=2),
            file_name="polylungai_results.json",
            mime="application/json"
        )
    except Exception as exc:
        st.error(f"API call failed: {exc}")

if uploaded is not None:
    st.info("Image received. Current mock flow uses metadata only; image model hook is ready for Phase 2.")
