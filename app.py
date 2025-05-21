# app.py (Updated to Match Latest Video Processing Integration)

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

st.set_page_config(page_title="AeroAI - AI Solar Panel Inspection", layout="wide")

PANEL_MODEL_PATH = "models/yolov8_panel.pt"
ANOMALY_MODEL_PATH = "models/yolov5_anomaly.pt"

st.sidebar.markdown("## 🚀 Loading Models...")
panel_model, anomaly_model = load_models(PANEL_MODEL_PATH, ANOMALY_MODEL_PATH)
st.sidebar.success("✅ Models Loaded Successfully!")

st.image("assets/aeroai_logo.png", width=200)
st.markdown("""
<style>
.big-title { font-size:48px; font-weight:bold; color:#22D3EE; }
.sub-title { font-size:24px; color:#64748B; }
</style>
""", unsafe_allow_html=True)
st.markdown('<div class="big-title">Welcome to AeroAI</div>', unsafe_allow_html=True)
st.markdown('<div class="sub-title">AI-powered Solar Panel Inspection Platform</div>', unsafe_allow_html=True)

# Tabs
tabs = st.tabs(["🏠 Home", "📤 Upload Media", "🖼️ Combined Result", "📊 Dashboard", "💰 Cost Estimation"])

    # Home
with tabs[0]:
    st.header("About AeroAI")
    st.write("""
    AeroAI revolutionizes solar farm maintenance with drone-powered, AI-based inspections.
    Upload images, videos, or connect your live drone feed to start analyzing your solar assets.
    """)

# Upload Media
with tabs[1]:
    st.header("Upload Your Media for Inspection")
    uploaded_files = st.file_uploader("Upload Images or Video Files", accept_multiple_files=True)

    if uploaded_files:
        for uploaded_file in uploaded_files:
            st.markdown(f"### Processing `{uploaded_file.name}`...")

            if uploaded_file.name.lower().endswith(('.mp4', '.mov', '.avi')):
                st.markdown("🎥 Detected video file. Running full video inspection...")
                panel_path, anomaly_path = process_video_file(uploaded_file, panel_model, ANOMALY_MODEL_PATH)

                if panel_path:
                    st.video(str(panel_path))
                if anomaly_path:
                    st.video(str(anomaly_path), format="video/mp4")
                st.markdown("✅ Video Processing Complete")

            else:
                panel_image_path, anomaly_image_path = process_image_file(uploaded_file, panel_model, ANOMALY_MODEL_PATH)

                # Safely display panel image
                if panel_image_path and Path(panel_image_path).exists():
                    st.image(str(panel_image_path), caption="Panel Detection", use_container_width=True)
                    st.session_state[f'panel_image_{uploaded_file.name}'] = str(panel_image_path)
                else:
                    st.warning(f"⚠️ Panel detection image not found for {uploaded_file.name}.")

                # Safely display anomaly image
                if anomaly_image_path and Path(anomaly_image_path).exists():
                    st.image(str(anomaly_image_path), caption="Anomaly Detection", use_container_width=True)
                    st.session_state[f'anomaly_image_{uploaded_file.name}'] = str(anomaly_image_path)
                else:
                    st.warning(f"⚠️ Anomaly detection image not found or not generated for {uploaded_file.name}.")

                st.markdown("✅ Image Processing Complete")

                # Match label filenames even if .png or .jpeg
                image_stem = Path(uploaded_file.name).stem
                panel_label_candidates = sorted(glob.glob(f"processed/panel*/labels/{image_stem}.txt"), reverse=True)
                anomaly_label_candidates = sorted(glob.glob(f"processed/anomaly*/labels/{image_stem}.txt"), reverse=True)

                panel_label_path = Path(panel_label_candidates[0]) if panel_label_candidates else None
                anomaly_label_path = Path(anomaly_label_candidates[0]) if anomaly_label_candidates else None

                if panel_label_path and panel_label_path.exists() and anomaly_label_path and anomaly_label_path.exists():
                    panel_class_map = {0: 'panel'}
                    anomaly_class_map = {0: 'cracked', 1: 'dusty', 2: 'normal'}

                    panel_boxes = parse_yolo_labels(panel_label_path, panel_class_map, panel_image_path)
                    anomaly_boxes = parse_yolo_labels(anomaly_label_path, anomaly_class_map, anomaly_image_path)

                    panel_anomaly_map = link_anomalies_to_panels(panel_boxes, anomaly_boxes)
                    st.session_state[f'panel_anomaly_map_{uploaded_file.name}'] = panel_anomaly_map
                else:
                    st.warning(f"Label files not found for {uploaded_file.name}. Skipping detailed analysis.")

# Combined Result
with tabs[2]:
    st.header("🖼️ Combined Result Viewer")

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

    # Video-based combined result
    summary_keys = [k for k in st.session_state if k.startswith('panel_anomaly_map_') and k.endswith('_summary')]
    if summary_keys:
        st.subheader("🎥 Video Inspection Results")

        for summary_key in summary_keys:
            video_stem = summary_key.replace("panel_anomaly_map_", "").replace("_summary", "")
            st.markdown(f"**Video ID:** `{video_stem}`")

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

# Dashboard and Cost Estimation tabs remain unchanged and use st.session_state as in original code
with tabs[3]:
    st.header("📊 Inspection Dashboard")

    # ✅ Filter: allow image-based or summary-level video keys only
    map_keys = [
        key for key in st.session_state
        if key.startswith('panel_anomaly_map_') and (
            key.endswith('_summary') or
            ('_frame' not in key and '_summary' not in key)
        )
    ]

    if not map_keys:
        st.info("No inspection data available. Please complete an inspection first.")
    else:
        total_panels = 0
        total_normal = 0
        total_dusty = 0
        total_cracked = 0

        st.subheader("🖼️ Inspection Breakdown")

        for key in map_keys:
            filename = key.replace("panel_anomaly_map_", "").replace("_summary", "")
            panel_anomaly_map = st.session_state[key]

            count_normal = 0
            count_dusty = 0
            count_cracked = 0

            for anomalies in panel_anomaly_map.values():
                # Prioritize: cracked > dusty > normal
                if 'cracked' in anomalies:
                    count_cracked += 1
                elif 'dusty' in anomalies:
                    count_dusty += 1
                elif 'normal' in anomalies or 'Normal' in anomalies:
                    count_normal += 1

            total_panels += len(panel_anomaly_map)
            total_normal += count_normal
            total_dusty += count_dusty
            total_cracked += count_cracked

            with st.expander(f"📄 {filename}"):
                st.markdown(f"- Total Panels: **{len(panel_anomaly_map)}**")
                st.markdown(f"- ✅ Normal: **{count_normal}**")
                st.markdown(f"- 🟠 Dusty: **{count_dusty}**")
                st.markdown(f"- 🔴 Cracked: **{count_cracked}**")

        # 🟢 Global summary
        st.subheader("📊 Aggregate Summary")
        st.write(f"**Total Panels Detected:** {total_panels}")
        st.write(f"**Dusty Panels:** {total_dusty}")
        st.write(f"**Cracked Panels:** {total_cracked}")
        st.write(f"**Normal Panels:** {total_normal}")

        # 📊 Pie Chart
        import matplotlib.pyplot as plt
        fig, ax = plt.subplots()
        ax.pie(
            [total_dusty, total_cracked, total_normal],
            labels=['Dusty', 'Cracked', 'Normal'],
            autopct='%1.1f%%',
            colors=['orange', 'red', 'green']
        )
        st.pyplot(fig)

        # 📤 CSV Export
        import pandas as pd
        report_data = {
            "Condition": ["Dusty", "Cracked", "Normal"],
            "Count": [total_dusty, total_cracked, total_normal]
        }
        report_df = pd.DataFrame(report_data)
        csv = report_df.to_csv(index=False).encode('utf-8')
        st.download_button(
            label="📥 Download CSV Report",
            data=csv,
            file_name='inspection_report.csv',
            mime='text/csv',
        )


with tabs[4]:
    st.header("💰 Cost Estimation Report")

    map_keys = [key for key in st.session_state if key.startswith('panel_anomaly_map_')]

    if not map_keys:
        st.info("No inspection data available. Please upload media first.")
    else:
        # Cost configuration
        COST_CLEANING = 5     # $5 per dusty panel
        COST_REPLACEMENT = 75 # $75 per cracked panel

        st.subheader("Per-Image Cost Breakdown")

        total_cost = 0
        total_panels = 0
        total_dusty = 0
        total_cracked = 0
        cost_rows = []

        for key in map_keys:
            filename = key.replace("panel_anomaly_map_", "")
            panel_anomaly_map = st.session_state[key]

            count_dusty = 0
            count_cracked = 0

            for anomalies in panel_anomaly_map.values():
                if 'dusty' in anomalies:
                    count_dusty += 1
                if 'cracked' in anomalies:
                    count_cracked += 1

            image_cost = (count_dusty * COST_CLEANING) + (count_cracked * COST_REPLACEMENT)
            total_cost += image_cost
            total_panels += len(panel_anomaly_map)
            total_dusty += count_dusty
            total_cracked += count_cracked

            with st.expander(f"🖼️ {filename}"):
                st.markdown(f"- 🧼 Dusty Panels: **{count_dusty}** → ${count_dusty * COST_CLEANING}")
                st.markdown(f"- 🔧 Cracked Panels: **{count_cracked}** → ${count_cracked * COST_REPLACEMENT}")
                st.markdown(f"**Estimated Repair Cost for {filename}:** `${image_cost}`")

            cost_rows.append([filename, count_dusty, count_cracked, image_cost])

        # Summary Table
        st.subheader("🔢 Total Cost Summary")
        st.write(f"**Total Panels Inspected:** {total_panels}")
        st.write(f"**Total Dusty Panels:** {total_dusty}")
        st.write(f"**Total Cracked Panels:** {total_cracked}")
        st.success(f"💰 **Total Estimated Cost:** `${total_cost}`")
        # 🖼️ Combined Result Visualizations
        st.subheader("🖼️ Combined Panel + Anomaly Visuals")

        for key in map_keys:
            filename = key.replace("panel_anomaly_map_", "")
            image_stem = Path(filename).stem
            combined_path = Path(f"processed/{image_stem}/combined.jpg")

            if combined_path.exists():
                st.image(str(combined_path), caption=f"Combined Detection: {filename}", use_container_width=True)
            else:
                st.warning(f"Combined image not found for: {filename}")


        # Exportable DataFrame
        import pandas as pd
        df_cost = pd.DataFrame(cost_rows, columns=["Image", "Dusty", "Cracked", "Estimated Cost"])
        csv = df_cost.to_csv(index=False).encode('utf-8')

        st.download_button(
            label="📥 Download Cost Estimate CSV",
            data=csv,
            file_name='cost_estimate_report.csv',
            mime='text/csv'
        )
