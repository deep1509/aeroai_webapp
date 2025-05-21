# app.py (Modified for Single Modern Sidebar Button)
from pages.costEstimation_page import render_cost_estimation
from pages.dashboard_page import render_dashboard

import streamlit as st
from aero_utils import (
    load_models,
    process_image_file,
    parse_yolo_labels,
    link_anomalies_to_panels,
    process_video_file
)
import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path
import glob



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

# 🚀 Page Configuration
st.set_page_config(page_title="AeroAI - AI Solar Panel Inspection", layout="wide")

# 🧭 Sidebar Navigation
st.sidebar.title("⚙️ AeroAI")
st.sidebar.markdown("### 🌐 Navigation")

# 🖌️ Load custom CSS from external file
with open("style.css") as f:
    st.markdown(f"<style>{f.read()}</style>", unsafe_allow_html=True)

# --- MODIFICATION START ---
# Use st.sidebar.radio for navigation
# This will create a single set of radio buttons, styled by your CSS
selected_page_label = st.sidebar.radio(
    "Go to",
    list(PAGES.keys()), # Options for the radio button are the page labels
    index=list(PAGES.keys()).index(st.session_state.page), # Set initial selection
    key="sidebar_navigation_radio" # Unique key for the radio button
)

# Update session state based on radio button selection
if selected_page_label:
    st.session_state.page = selected_page_label
# --- MODIFICATION END ---

# 📦 Load Models
PANEL_MODEL_PATH = "models/yolov8_panel.pt"
ANOMALY_MODEL_PATH = "models/yolov5_anomaly.pt"
panel_model, anomaly_model = load_models(PANEL_MODEL_PATH, ANOMALY_MODEL_PATH)
st.sidebar.success("✅ Models Loaded Successfully!")

# 🔀 Routing
page = st.session_state.page

# 🔽 PAGE ROUTES BELOW HERE (Home, Upload Media, Combined Result, Dashboard, Cost Estimation)
# [Insert your previously working logic per page here unchanged]

# ──────────────────────────────────────────────────────────────────────
# 🧭 ROUTE: Home
if page == "🏠 Home":
    st.image("assets/aeroai_logo.png", width=200)
    st.markdown("""
    <style>
    .big-title {
        font-size:48px;
        font-weight:bold;
        color:#22D3EE;
        margin-top: 1rem;
    }
    .sub-title {
        font-size:24px;
        color:#94A3B8;
        margin-bottom: 2rem;
    }
    </style>
    """, unsafe_allow_html=True)
    st.markdown('<div class="big-title">Welcome to AeroAI</div>', unsafe_allow_html=True)
    st.markdown('<div class="sub-title">AI-powered Solar Panel Inspection Platform</div>', unsafe_allow_html=True)
    st.markdown("AeroAI revolutionizes solar farm maintenance with drone-powered, AI-based inspections.")
    st.markdown("Upload images, videos, or connect your live drone feed to start analyzing your solar assets.")
    col1, col2 = st.columns(2)
    with col1:
        st.button("📤 Start Inspection")
    with col2:
        st.button("📈 View Reports")

# ──────────────────────────────────────────────────────────────────────
# 🧭 ROUTE: Upload Media (will implement fully in Step 3)
elif page == "📤 Upload Media":
    st.title("📤 Upload Inspection Media")
    st.info("Upload your inspection files here (images or videos).")

    uploaded_files = st.file_uploader("Upload Images or Video Files", accept_multiple_files=True)

    if uploaded_files:
        for uploaded_file in uploaded_files:
            with st.spinner(f"⏳ Processing `{uploaded_file.name}`... Please wait."):
                if uploaded_file.name.lower().endswith(('.mp4', '.mov', '.avi')):
                    # Video file
                    st.write("🎥 Detected video file. Running inspection...")

                    panel_path, anomaly_path = process_video_file(uploaded_file, panel_model, ANOMALY_MODEL_PATH)

                    # Don't show video preview here
                    st.session_state[f'anomaly_video_{Path(uploaded_file.name).stem}'] = str(anomaly_path)
                    st.session_state[f'panel_video_{Path(uploaded_file.name).stem}'] = str(panel_path)
                    st.success(f"✅ Video Processing Complete: {uploaded_file.name}")

                else:
                    # Image file
                    st.write("🖼️ Detected image file. Running analysis...")

                    panel_image_path, anomaly_image_path = process_image_file(uploaded_file, panel_model, ANOMALY_MODEL_PATH)

                    # Store in session only; do not display
                    st.session_state[f'panel_image_{uploaded_file.name}'] = str(panel_image_path)
                    st.session_state[f'anomaly_image_{uploaded_file.name}'] = str(anomaly_image_path)

                    image_stem = Path(uploaded_file.name).stem
                    panel_label_candidates = sorted(glob.glob(f"processed/panel*/labels/{image_stem}.txt"), reverse=True)
                    anomaly_label_candidates = sorted(glob.glob(f"processed/anomaly*/labels/{image_stem}.txt"), reverse=True)

                    panel_label_path = Path(panel_label_candidates[0]) if panel_label_candidates else None
                    anomaly_label_path = Path(anomaly_label_candidates[0]) if anomaly_label_candidates else None

                    if panel_label_path and anomaly_label_path and panel_label_path.exists() and anomaly_label_path.exists():
                        panel_class_map = {0: 'panel'}
                        anomaly_class_map = {0: 'cracked', 1: 'dusty', 2: 'normal'}

                        panel_boxes = parse_yolo_labels(panel_label_path, panel_class_map, panel_image_path)
                        anomaly_boxes = parse_yolo_labels(anomaly_label_path, anomaly_class_map, anomaly_image_path)

                        panel_anomaly_map = link_anomalies_to_panels(panel_boxes, anomaly_boxes)
                        st.session_state[f'panel_anomaly_map_{uploaded_file.name}'] = panel_anomaly_map
                    else:
                        st.warning(f"⚠️ Labels not found for {uploaded_file.name}. Skipping mapping.")

                    st.success(f"✅ Image Processing Complete: {uploaded_file.name}")

# ──────────────────────────────────────────────────────────────────────
# 🧭 ROUTE: Combined Result (to be redesigned in Step 4)
elif page == "🖼️ Combined Result":
    st.title("🖼️ Combined Results")
    st.info("Combined view of detected panels and anomalies.")

    # Image-based results
    panel_keys = [key for key in st.session_state if key.startswith('panel_image_')]
    if panel_keys:
        for panel_key in panel_keys:
            image_name = panel_key.replace('panel_image_', '')
            panel_path = st.session_state[panel_key]
            anomaly_key = f'anomaly_image_{image_name}'
            anomaly_path = st.session_state.get(anomaly_key, None)

            st.subheader(f"🖼️ Image: {image_name}")
            st.image(panel_path, caption="Panel Detection", use_container_width=True)
            if anomaly_path:
                st.image(anomaly_path, caption="Anomaly Detection", use_container_width=True)

            panel_anomaly_key = f'panel_anomaly_map_{image_name}'
            if panel_anomaly_key in st.session_state:
                st.write("Detected Panel-Anomaly Mapping:")
                st.json(st.session_state[panel_anomaly_key])
            else:
                st.warning("Panel-Anomaly mapping not available for this image.")

    # Video-based combined result summary
    summary_keys = [k for k in st.session_state if k.startswith('panel_anomaly_map_') and k.endswith('_summary')]
    if summary_keys:
        st.subheader("🎥 Video Inspection Results")
        for summary_key in summary_keys:
            video_stem = summary_key.replace("panel_anomaly_map_", "").replace("_summary", "")
            st.markdown(f"**Video ID:** {video_stem}")
            video_key = f'anomaly_video_{video_stem}'
            thumb_key = f'anomaly_video_frame_{video_stem}'

            if video_key in st.session_state:
                st.video(st.session_state[video_key], format="video/mp4")
                thumb_path = st.session_state.get(thumb_key)
                if thumb_path and isinstance(thumb_path, (str, Path)) and Path(str(thumb_path)).exists():
                    st.image(str(thumb_path), caption="Preview Frame", use_container_width=True)
                st.write("Detected Panel-Anomaly Mapping (summary):")
                st.json(st.session_state[summary_key])
            else:
                if not panel_keys:
                    st.info("No results found yet. Please upload images or videos first.")

# ──────────────────────────────────────────────────────────────────────
# 🧭 ROUTE: Dashboard
# 🧭 ROUTE: Dashboard
elif page == "📊 Dashboard":
    render_dashboard()

# ──────────────────────────────────────────────────────────────────────
# 🧭 ROUTE: Cost Estimation
elif page == "💰 Cost Estimation":
    render_cost_estimation()