import streamlit as st
# 🚀 Page Configuration
st.set_page_config(page_title="AeroAI - AI Solar Panel Inspection", layout="wide")
from components.costEstimation_page import render_cost_estimation
from components.dashboard_page import render_dashboard
from components.combined_results import show_combined_results   
from components.upload_media import show_upload_page, reset_inspection_data 
from components.home_page import show_home_page


from aero_utils import (
    load_models,
    process_image_file,
    parse_yolo_labels,
    link_anomalies_to_panels,
    process_video_file
)
#
import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path
import glob
import base64

with open("style.css") as f:
    st.markdown(f"<style>{f.read()}</style>", unsafe_allow_html=True)


# 🗂️ App Pages
PAGES = {
    "🏠 Home": "home",
    "📤 Upload Media": "upload",
    "🖼️ Combined Result": "combined",
    "📊 Dashboard": "dashboard",
    "💰 Cost Estimation": "cost"
}

# ✅ Initialize session state before any logic
if "page" not in st.session_state:
    st.session_state["page"] = "🏠 Home"

#dfsd

# 🧭 Sidebar Navigation
st.sidebar.markdown(
    """
    <style>
    .sidebar-header {
      font-size: 4rem;
      font-weight: 700;
      color: #22D3EE;      /* your cyan accent */
      margin-bottom: 1rem;
      text-align: center;
    }
    </style>
    <div class="sidebar-header">AeroAI</div>
    """,
    unsafe_allow_html=True
)



# 🖌️ Load custom CSS from external file
with open("style.css") as f:
    st.markdown(f"<style>{f.read()}</style>", unsafe_allow_html=True)

# --- MODIFICATION START ---
# Replace radio navigation with buttons
for label in PAGES.keys():
    if st.sidebar.button(label):
        st.session_state.page = label
# --- MODIFICATION END ---

# 📦 Load Models
PANEL_MODEL_PATH = "models/yolov8_panel.pt"
ANOMALY_MODEL_PATH = "models/yolov5_anomaly.pt"

st.sidebar.markdown("## 🚀 Loading Models...")
panel_model, anomaly_model = load_models(PANEL_MODEL_PATH, ANOMALY_MODEL_PATH)
st.sidebar.success("✅ Models Loaded Successfully!")

# 🔀 Routing
page = st.session_state.page

# ──────────────────────────────────────────────────────────────────────
# helper to set a base64 background image
def set_home_background(png_path: str):
    with open(png_path, "rb") as img_file:
        b64 = base64.b64encode(img_file.read()).decode()
    css = f"""
    <style>
    [data-testid="stAppViewContainer"] {{
        background: url("data:image/png;base64,{b64}") no-repeat center center fixed;
        background-size: cover;
    }}
    [data-testid="stAppViewContainer"]::before {{
        content: "";
        position: fixed;
        top: 0; left: 0; right: 0; bottom: 0;
        background: rgba(15,23,42,0.6);
        pointer-events: none;
    }}
    </style>
    """
    st.markdown(css, unsafe_allow_html=True)

# 🧭 ROUTE: Home
if page == "🏠 Home":
    show_home_page()

# ──────────────────────────────────────────────────────────────────────
# 🧭 ROUTE: Upload Media
elif page == "📤 Upload Media":
    show_upload_page(panel_model, ANOMALY_MODEL_PATH)


# ──────────────────────────────────────────────────────────────────────
#Combined Results 
elif page == "🖼️ Combined Result":
    show_combined_results()
    #\


# ──────────────────────────────────────────────────────────────────────
# 🧭 ROUTE: Dashboard
elif page == "📊 Dashboard":
    render_dashboard()

# ──────────────────────────────────────────────────────────────────────
# 🧭 ROUTE: Cost Estimation
elif page == "💰 Cost Estimation":
    render_cost_estimation()
