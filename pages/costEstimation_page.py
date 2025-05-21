import streamlit as st
import pandas as pd
from datetime import datetime
from pathlib import Path

def render_cost_estimation():
    st.title("💰 Cost Estimation Summary")
    st.info("Total inspection maintenance cost overview with key insights.")

    map_keys = [key for key in st.session_state if key.startswith('panel_anomaly_map_')]

    if not map_keys:
        st.warning("⚠️ No inspection data available.")
        return

    COST_CLEANING = 5
    COST_REPLACEMENT = 75
    total_cost = 0
    total_dusty = 0
    total_cracked = 0
    cost_rows = []

    for key in map_keys:
        filename = key.replace("panel_anomaly_map_", "")
        panel_anomaly_map = st.session_state[key]
        dusty = sum('dusty' in anomalies for anomalies in panel_anomaly_map.values())
        cracked = sum('cracked' in anomalies for anomalies in panel_anomaly_map.values())
        file_cost = (dusty * COST_CLEANING) + (cracked * COST_REPLACEMENT)
        total_cost += file_cost
        total_dusty += dusty
        total_cracked += cracked
        cost_rows.append([filename, dusty, cracked, file_cost])

    df = pd.DataFrame(cost_rows, columns=["File", "Dusty", "Cracked", "Cost"])
    most_expensive = df.sort_values("Cost", ascending=False).iloc[0]

    # 🚨 Cost Impact Badge
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
    st.markdown("""
        <div style="text-align: center; padding: 2rem 0;">
            <h1 style="font-size: 48px; color: #FACC15;">💰 ${:,}</h1>
            <h3 style="color: #94a3b8;">Total Maintenance Cost</h3>
        </div>
    """.format(total_cost), unsafe_allow_html=True)

    # 🏷️ Impact Rating
    st.markdown(f"<h5 style='text-align:center; color:{color};'>{impact}</h5>", unsafe_allow_html=True)

    st.markdown("---")

    # 🥇 Most Expensive File
    st.subheader("📸 Most Expensive File")
    st.markdown(f"**{most_expensive['File']}** → ${int(most_expensive['Cost'])}")

    # 💡 Savings Insight
    cleaning_only_cost = total_dusty * COST_CLEANING
    savings_loss = total_cost - cleaning_only_cost

    st.subheader("💡 Cracked Panel Cost Impact")
    st.markdown(f"If all panels were just dusty, estimated cost would be **${cleaning_only_cost}**")
    st.markdown(f"🔧 **Extra cost due to cracks:** `${savings_loss}`")

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
    
    st.markdown("### 💼 Cost Matrix (Current Rates)")
    col1, col2 = st.columns(2)

    with col1:
        st.markdown("""
        <div style="background-color:#1e293b; padding:1.5rem; border-radius:12px; color:#e2e8f0; text-align:center;">
            <h4>🧼 Cleaning</h4>
            <p>Per Dusty Panel</p>
            <h2 style="color:#facc15;">$5</h2>
        </div>
        """, unsafe_allow_html=True)

    with col2:
        st.markdown("""
        <div style="background-color:#1e293b; padding:1.5rem; border-radius:12px; color:#e2e8f0; text-align:center;">
            <h4>🔧 Replacement</h4>
            <p>Per Cracked Panel</p>
            <h2 style="color:#f87171;">$75</h2>
        </div>
        """, unsafe_allow_html=True)



    # ⏱ Timestamp
    st.caption(f"🕒 Last Updated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
