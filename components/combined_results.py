import streamlit as st
from pathlib import Path

def show_combined_results():
    st.markdown("""
    <h1 style='color:#22D3EE; font-weight:800; font-size:2.5rem;'>
    🖼️ Combined Results
    </h1>
    """, unsafe_allow_html=True)

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
    .image-pair {
        margin-bottom: 2rem;
    }
    .caption {
        text-align: center;
        color: #CBD5E1 !important;
        font-weight: 600;
    }
</style>
    """, unsafe_allow_html=True)

    panel_keys = [key for key in st.session_state if key.startswith('panel_image_') or key.startswith('panel_video_')]

    if not panel_keys:
        st.info("No results found yet. Please upload images or videos first.")
        return

    for panel_key in panel_keys:
        is_video = panel_key.startswith('panel_video_')
        image_name = panel_key.replace('panel_image_', '').replace('panel_video_', '')
        panel_path = st.session_state[panel_key]
        anomaly_key = f"anomaly_image_{image_name}" if not is_video else f"anomaly_video_{image_name}"
        anomaly_path = st.session_state.get(anomaly_key)

        st.subheader(f"🖼️ {image_name}")
        col1, col2 = st.columns(2)

        with col1:
            if panel_path and Path(panel_path).exists():
                if is_video:
                    st.video(str(panel_path))
                else:
                    st.image(str(panel_path), use_container_width=True)
                st.markdown("<div class='caption'>Panel Detection</div>", unsafe_allow_html=True)
            else:
                st.warning("⚠️ Panel result not found.")

        with col2:
            if anomaly_path and Path(anomaly_path).exists():
                if is_video:
                    st.video(str(anomaly_path))
                else:
                    st.image(str(anomaly_path), use_container_width=True)
                st.markdown("<div class='caption'>Anomaly Detection</div>", unsafe_allow_html=True)
            else:
                st.warning("⚠️ No anomaly detection result available.")

    st.markdown("---")
