import streamlit as st
import pandas as pd
from datetime import datetime
from pathlib import Path

from theme import apply_dark_theme
apply_dark_theme()

def get_panel_cost_matrix(panel_type):
    """
    Returns a dictionary of cleaning and replacement costs based on the solar panel type.
    """
    cost_matrix = {
        "Monocrystalline": {"cleaning": 5, "replacement": 90},
        "Polycrystalline": {"cleaning": 5, "replacement": 75},
        "Thin-Film": {"cleaning": 5, "replacement": 60}
    }
    # Return costs for the selected panel type, or default if not found
    return cost_matrix.get(panel_type, {"cleaning": 5, "replacement": 75})

def render_cost_estimation():
    """
    Renders the cost estimation summary for solar panel maintenance.
    Calculates total costs based on anomalies detected in inspection data.
    """
    st.markdown("""
    <h1 style='color:#22D3EE; font-weight:800; font-size:2.5rem;'>
   💰 Cost Estimation Summary
    </h1>
    """, unsafe_allow_html=True)

    # Custom CSS to ensure text visibility against dark background
    st.markdown("""
    <style>
    /* Change the color of the selectbox label to a light color */
    .stSelectbox > label {
        color: #E2E8F0 !important; /* Light text color for better contrast */
    }
    /* Ensure the caption text is visible */
    .caption {
        text-align: center;
        color: #94a3b8 !important; /* Lighter grey for captions */
        font-weight: 600;
    }
    /* General background and text color, already handled by apply_dark_theme but kept for clarity */
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
    </style>
    """, unsafe_allow_html=True)


    st.info("Total inspection maintenance cost overview with key insights.")
    
    # Select panel type and apply rates
    panel_type = st.selectbox("🔧 Select Solar Panel Type", ["Monocrystalline", "Polycrystalline", "Thin-Film"])
    rates = get_panel_cost_matrix(panel_type)
    COST_CLEANING = rates["cleaning"]
    COST_REPLACEMENT = rates["replacement"]

    # # Initialize session state if not already done, for testing purposes
    # if 'panel_anomaly_map_example1.jpg' not in st.session_state:
    #     st.session_state['panel_anomaly_map_example1.jpg'] = {
    #         'panel_1': ['dusty'],
    #         'panel_2': ['cracked'],
    #         'panel_3': ['dusty', 'cracked']
    #     }
    #     st.session_state['panel_anomaly_map_example2.jpg'] = {
    #         'panel_1': ['dusty'],
    #         'panel_2': [],
    #         'panel_3': ['cracked']
    #     }

    map_keys = [
        key for key in st.session_state
        if key.startswith('panel_anomaly_map_') and (
            key.endswith('_summary') or ('_frame' not in key and '_summary' not in key)
        )
    ]


    if not map_keys:
        st.warning("⚠️ No inspection data available. Please upload images and run detection to see cost estimates.")
        return

    total_cost = 0
    total_dusty = 0
    total_cracked = 0
    cost_rows = []

    for key in map_keys:
        filename = key.replace("panel_anomaly_map_", "").replace("_summary", "")
        panel_anomaly_map = st.session_state[key]
        # Count 'dusty' and 'cracked' anomalies for each file
        dusty = sum('dusty' in anomalies for anomalies in panel_anomaly_map.values())
        cracked = sum('cracked' in anomalies for anomalies in panel_anomaly_map.values())
        
        # Calculate cost for the current file
        file_cost = (dusty * COST_CLEANING) + (cracked * COST_REPLACEMENT)
        
        # Accumulate total costs and anomaly counts
        total_cost += file_cost
        total_dusty += dusty
        total_cracked += cracked
        
        # Add row to the cost_rows list for DataFrame creation
        cost_rows.append([filename, dusty, cracked, file_cost])

    # Create a DataFrame from the collected cost data
    df = pd.DataFrame(cost_rows, columns=["File", "Dusty", "Cracked", "Cost"])
    
    # Get the most expensive file if the DataFrame is not empty
    most_expensive = None
    if not df.empty:
        most_expensive = df.sort_values("Cost", ascending=False).iloc[0]

    # 🚨 Cost Impact Badge
    impact = ""
    color = ""
    if total_cost > 10000:
        impact = "🔴 Critical Cost Impact"
        color = "red"
    elif total_cost > 5000:
        impact = "🟡 Moderate Cost Impact"
        color = "orange"
    else:
        impact = "🟢 Low Cost Impact"
        color = "green"

    # 🔥 Big Centered Cost Display
    st.markdown(f"""
        <div style="text-align: center; padding: 2rem 0;">
            <h1 style="font-size: 48px; color: #FACC15;">💰 ${total_cost:,.0f}</h1>
            <h3 style="color: #94a3b8;">Total Maintenance Cost</h3>
        </div>
    """, unsafe_allow_html=True) # Use f-string for direct formatting

    # 🏷️ Impact Rating
    st.markdown(f"<h5 style='text-align:center; color:{color};'>{impact}</h5>", unsafe_allow_html=True)

    st.markdown("---")

    # 🥇 Most Expensive File
    if most_expensive is not None:
        st.subheader("📸 Most Expensive File")
        st.markdown(f"**{most_expensive['File']}** → ${int(most_expensive['Cost'])}")
    else:
        st.subheader("📸 Most Expensive File")
        st.info("No files to analyze for cost.")

    # 💡 Savings Insight
    cleaning_only_cost = total_dusty * COST_CLEANING
    savings_loss = total_cost - cleaning_only_cost

    st.subheader("💡 Cracked Panel Cost Impact")
    st.markdown(f"If all panels were just dusty, estimated cost would be **${cleaning_only_cost:,.0f}**")
    st.markdown(f"🔧 **Extra cost due to cracks:** `${savings_loss:,.0f}`")

    # 📄 Full Breakdown Table
    st.markdown("### 📂 Full File Breakdown")
    st.dataframe(df, use_container_width=True)

    # 📥 Download CSV
    st.download_button(
        "📥 Download Cost Report",
        data=df.to_csv(index=False).encode("utf-8"),
        file_name="cost_report.csv",
        mime="text/csv"
    )
    
     # 💼 Pricing Matrix (Dynamic by panel type)
    st.markdown("### 💼 Cost Matrix (Current Rates)")
    st.markdown(f"<p style='color:#CBD5E1; font-weight:500;'>Selected Panel Type: <strong>{panel_type}</strong></p>", unsafe_allow_html=True)
    col1, col2 = st.columns(2)

    with col1:
        st.markdown(f"""
        <div class="matrix-panel-box" style="background-color:#1e293b; padding:1.5rem; border-radius:12px; color:#e2e8f0; text-align:center;">
                        <h4>🧼 Cleaning</h4>
            <p>Per Dusty Panel</p>
            <h2 style="color:#facc15;">${COST_CLEANING}</h2>
        </div>
        """, unsafe_allow_html=True)

    with col2:
        st.markdown(f"""
        <div class="matrix-panel-box" style="background-color:#1e293b; padding:1.5rem; border-radius:12px; color:#e2e8f0; text-align:center;">
            <h4>🔧 Replacement</h4>
            <p>Per Cracked Panel</p>
            <h2 style="color:#f87171;">${COST_REPLACEMENT}</h2>
        </div>
        """, unsafe_allow_html=True)


    # ⏱ Timestamp
    st.caption(f"🕒 Last Updated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
