import streamlit as st
import matplotlib.pyplot as plt
import pandas as pd
from datetime import datetime
import seaborn as sns
sns.set_theme(style="dark")  # dark-compatible grid style
plt.style.use("dark_background")  # ensure matplotlib matches
colors = sns.color_palette("dark")   # Other good options below



def render_dashboard():
    st.title("📊 Inspection Dashboard")
    st.info("Visual summaries and analytics will appear here.")

    # Filter inspection data from session
    map_keys = [
        key for key in st.session_state
        if key.startswith('panel_anomaly_map_') and (
            key.endswith('_summary') or ('_frame' not in key and '_summary' not in key)
        )
    ]

    if not map_keys:
        st.warning("⚠️ No inspection data available. Please complete an inspection first.")
        return

    # Initialize counters
    total_panels = 0
    total_normal = 0
    total_dusty = 0
    total_cracked = 0
    table_rows = []

    # Process each uploaded file's result
    for key in map_keys:
        filename = key.replace("panel_anomaly_map_", "").replace("_summary", "")
        panel_anomaly_map = st.session_state[key]
        norm, dusty, cracked = 0, 0, 0

        for anomalies in panel_anomaly_map.values():
            if 'cracked' in anomalies:
                cracked += 1
            elif 'dusty' in anomalies:
                dusty += 1
            elif 'normal' in anomalies or 'Normal' in anomalies:
                norm += 1

        total_panels += len(panel_anomaly_map)
        total_normal += norm
        total_dusty += dusty
        total_cracked += cracked

        table_rows.append([filename, len(panel_anomaly_map), norm, dusty, cracked])

    # ──────────── 🔢 KPI Cards ────────────
    st.markdown("### 📌 Key Metrics")
    col1, col2, col3, col4 = st.columns(4)
    col1.metric("Total Panels", total_panels)
    col2.metric("✅ Normal", total_normal)
    col3.metric("🟠 Dusty", total_dusty)
    col4.metric("🔴 Cracked", total_cracked)

    st.markdown("---")

    # ──────────── 📊 Charts Row ────────────
    st.markdown("### 📊 Condition Breakdown")

    # Prepare DataFrame
    chart_df = pd.DataFrame({
        "Condition": ["Dusty", "Cracked", "Normal"],
        "Count": [total_dusty, total_cracked, total_normal]
    })

    col1, col2 = st.columns(2)

    # 🥧 Donut Chart with Legend
    with col1:
        st.markdown("#### 🥧 Anomaly Distribution")
        fig1, ax1 = plt.subplots()
        colors = ['#facc15', '#f87171', '#4ade80']  # yellow, red, green - vibrant for dark bg

        wedges, texts, autotexts = ax1.pie(
            chart_df["Count"],
            labels=chart_df["Condition"],
            colors=colors,
            autopct='%1.1f%%' if sum(chart_df["Count"]) > 0 else None,
            startangle=90,
            wedgeprops=dict(width=0.4, edgecolor='black')
        )
        ax1.axis('equal')
        ax1.legend(wedges, chart_df["Condition"], title="Condition", loc="center left", bbox_to_anchor=(1, 0, 0.5, 1))
        st.pyplot(fig1)
        plt.close(fig1)

    # 📊 Bar Chart with Dark Theme + Legend
    with col2:
        st.markdown("#### 📶 Panel Type Counts")
        fig2, ax2 = plt.subplots(figsize=(5, 4))
        bars = sns.barplot(
            x="Condition",
            y="Count",
            data=chart_df,
            palette=colors,
            ax=ax2
        )
        for bar in ax2.patches:
            height = bar.get_height()
            ax2.text(
                bar.get_x() + bar.get_width() / 2,
                height + 1,
                f'{int(height)}',
                ha='center',
                va='bottom',
                color='white',
                fontweight='bold'
            )
        ax2.set_title("Panel Conditions Count", fontsize=14, color='white')
        ax2.set_ylabel("Count", color='white')
        ax2.set_xlabel("", color='white')
        ax2.tick_params(colors='white')
        ax2.legend(
            handles=[plt.Rectangle((0, 0), 1, 1, color=c) for c in colors],
            labels=chart_df["Condition"].tolist(),  # ✅ Convert to list
            title="Condition",
            loc="upper right"
        )

        sns.despine(left=True, bottom=True)
        st.pyplot(fig2)
        plt.close(fig2)


    st.markdown("---")

    # ──────────── 📁 Per File Table ────────────
    st.markdown("### 📂 File-wise Summary")
    summary_df = pd.DataFrame(table_rows, columns=["File", "Total Panels", "Normal", "Dusty", "Cracked"])
    st.dataframe(summary_df, use_container_width=True)

    # ──────────── 📥 Export Button ────────────
    csv_data = summary_df.to_csv(index=False).encode("utf-8")
    st.download_button(
        "📥 Download Summary CSV",
        data=csv_data,
        file_name="inspection_summary.csv",
        mime="text/csv"
    )

    # ──────────── ⏱ Timestamp ────────────
    st.caption(f"🕒 Last Updated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
