# upload_media.py
import streamlit as st
from pathlib import Path
import glob
from aero_utils import (
    process_image_file,
    process_video_file,
    parse_yolo_labels,
    link_anomalies_to_panels
)

def reset_inspection_data():
    keys_to_clear = [key for key in st.session_state if any(
        key.startswith(prefix) for prefix in [
            'panel_image_', 'anomaly_image_', 'panel_anomaly_map_',
            'panel_video_', 'anomaly_video_', 'anomaly_video_frame_',
            'summary_temp_video'
        ])
    ]
    for key in keys_to_clear:
        del st.session_state[key]
    st.success("🔄 Inspection data reset.")

def show_upload_page(panel_model, anomaly_model_path):
    st.markdown("""
    <style>
    html, body, [data-testid="stAppViewContainer"] {
        background-color: #0F172A !important;
        color: #E2E8F0 !important;
    }
    [data-testid="stSidebar"] {
        background-color: #1E293B !important;
        color: #E2E8F0 !important;
    }
  reset-button-container {
    display: flex;
    justify-content: center;
    margin-top: 2rem;
}
.reset-button-container button {
    background-color: #dc2626 !important;
    color: white !important;
    font-weight: 600;
    font-size: 16px;
    padding: 0.75rem 2rem;
    border: none;
    border-radius: 8px;
    width: 240px;
    text-align: center;
    box-shadow: 0 2px 4px rgba(0, 0, 0, 0.2);
    transition: background-color 0.3s ease;
}
.reset-button-container button:hover {
    background-color: #dc2626 !important;
    filter: brightness(1.1);
    cursor: pointer;
}
    h1, h2, h3, h4 {
        color: #22D3EE;
    }
    p, span, div {
        color: #CBD5E1;
    }
    </style>
    """, unsafe_allow_html=True)

    st.markdown("""
        <h1 style='color:#F7931E; font-weight:800; font-size:2.5rem;'>
        📤 Upload Inspection Media
        </h1>
        """, unsafe_allow_html=True)

    st.info("Upload your inspection files here (images or videos).")

    uploaded_files = st.file_uploader("Upload Images or Video Files", accept_multiple_files=True)

    # Add this after file uploader
    col1, col2, col3 = st.columns([1, 0.5, 1])  # Center the button in the middle column
    with col2:
        if st.button("♻️ Reset Uploaded Data", key="reset_upload_btn"):
            reset_inspection_data()






    if st.session_state.get("reset_uploads"):
        reset_inspection_data()

    if uploaded_files:
        for uploaded_file in uploaded_files:
            with st.spinner(f"⏳ Processing `{uploaded_file.name}`... Please wait."):
                if uploaded_file.name.lower().endswith(('.mp4', '.mov', '.avi')):
                    st.write("🎥 Detected video file. Running inspection...")

                    panel_path, anomaly_path = process_video_file(uploaded_file, panel_model, anomaly_model_path)
                    st.session_state[f'anomaly_video_{Path(uploaded_file.name).stem}'] = str(anomaly_path)
                    st.session_state[f'panel_video_{Path(uploaded_file.name).stem}'] = str(panel_path)
                    st.success(f"✅ Video Processing Complete: {uploaded_file.name}")

                else:
                    st.write("🖼️ Detected image file. Running analysis...")

                    panel_image_path, anomaly_image_path = process_image_file(
                        uploaded_file, panel_model, anomaly_model_path)

                    st.session_state[f'panel_image_{uploaded_file.name}'] = str(panel_image_path)
                    st.session_state[f'anomaly_image_{uploaded_file.name}'] = str(anomaly_image_path)

                    image_stem = Path(uploaded_file.name).stem
                    panel_label_candidates = sorted(
                        glob.glob(f"processed/panel*/labels/{image_stem}.txt"), reverse=True
                    )
                    anomaly_label_candidates = sorted(
                        glob.glob(f"processed/anomaly*/labels/{image_stem}.txt"), reverse=True
                    )

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
