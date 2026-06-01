import os

import requests
import streamlit as st

API_URL = os.getenv("API_URL", "http://localhost:8000")

st.set_page_config(page_title="PolyLung Bridge AI", layout="wide")
st.title("PolyLung Bridge AI Dashboard")
st.caption("Module 1 + mock Module 2 bridge for polymer-resolved lung risk")

uploaded = st.file_uploader("Upload microscopy image", type=["png", "jpg", "jpeg", "tif", "tiff"])
exposure = st.selectbox("Exposure route", ["ingestion", "inhalation", "dermal"], index=0)
zip_input = st.text_input("Enter 5-digit ZIP Code", value="32501", max_chars=5)
poly_type = st.selectbox("Polymer Type", ["PVC", "PS", "PU", "PE", "PP", "PET", "Nylon", "Acrylic", "PC", "ABS"], index=0)
particle_cnt = st.number_input("Particle Count", min_value=0, value=120, step=1)

if st.button("Analyze"):
    if uploaded is None:
        st.warning("Please upload a microscopy image first before running the analysis.")
    else:
                
        data_payload = {
            "polyType": poly_type,          
            "particleCount": str(particle_cnt),      
            "exposRoute": exposure,
            "zipcode": zip_input,       
        }
        files_payload = {
            "file": (uploaded.name, uploaded.getvalue(), uploaded.type)
        }
        try:
            response = requests.post(
                f"{API_URL}/analyze", 
                data=data_payload, 
                files=files_payload, 
                timeout=15
            )
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
    st.info("Image received. Image upload analysis pipeline is currently offline; backend processing is using metadata inputs only.")