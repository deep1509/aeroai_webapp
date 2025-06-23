import streamlit as st
import matplotlib.pyplot as plt
import pandas as pd
from datetime import datetime
import seaborn as sns
sns.set_theme(style="dark")  # dark-compatible grid style
plt.style.use("dark_background")  # ensure matplotlib matches
colors = sns.color_palette("dark")   # Other good options below
from fpdf import FPDF
from io import BytesIO
from datetime import datetime




def render_dashboard():
    
    st.markdown("""
    <h1 style='color:#22D3EE; font-weight:800; font-size:2.5rem;'>
    Dashboard
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
            color: black !important;
            font-weight: 600;
        }
        </style>
    """, unsafe_allow_html=True)


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
    pdf_data = generate_dashboard_report_pdf(
    summary_df,
    total_panels=summary_df["Total Panels"].sum(),
    total_normal=summary_df["Normal"].sum(),
    total_dusty=summary_df["Dusty"].sum(),
    total_cracked=summary_df["Cracked"].sum()
    )

    st.download_button(
        label="📥 Download PDF Report",
        data=pdf_data,
        file_name="AeroAI_Inspection_Report.pdf",
        mime="application/pdf"
    )

# 🧾 TechFest-Ready PDF Dashboard Report for AeroAI
from fpdf import FPDF
from datetime import datetime
from io import BytesIO

class DashboardPDF(FPDF):
    def header(self):
        self.set_fill_color(15, 23, 42)
        self.rect(0, 0, 210, 25, 'F')
        self.image("assets/logo1.png", 10, 5, 15)
        self.set_text_color(255, 255, 255)
        self.set_font("Arial", "B", 14)
        self.set_y(8)
        self.cell(0, 10, "AeroAI - Inspection Dashboard Summary", ln=True, align="C")

    def footer(self):
        self.set_y(-15)
        self.set_font("Arial", "I", 8)
        self.set_text_color(180, 180, 180)
        self.cell(0, 10, f"Generated on {datetime.now().strftime('%Y-%m-%d %H:%M:%S')} | Page {self.page_no()}", align="C")

def generate_dashboard_report_pdf(summary_df, total_panels, total_normal, total_dusty, total_cracked):
    pdf = DashboardPDF()
    pdf.add_page()

    # Full page dark background
    pdf.set_fill_color(15, 23, 42)
    pdf.rect(0, 25, 210, 275, 'F')

    pdf.ln(20)
    pdf.set_font("Arial", "B", 16)
    pdf.set_text_color(255, 165, 0)
    pdf.cell(0, 10, "Panel Condition Summary", ln=True, align="C")
    pdf.ln(10)

    # Summary Cards
    x_positions = [15, 75, 135]
    labels = ["Total Panels", "Normal Panels", "Anomalous Panels"]
    values = [total_panels, total_normal, total_dusty + total_cracked]
    colors = [(224, 242, 254), (220, 252, 231), (254, 226, 226)]

    y_start = pdf.get_y()
    for i in range(3):
        pdf.set_xy(x_positions[i], y_start)
        pdf.set_fill_color(*colors[i])
        pdf.rect(x_positions[i], y_start, 60, 25, 'F')
        pdf.set_text_color(30, 41, 59)
        pdf.set_font("Arial", "B", 12)
        pdf.set_xy(x_positions[i], y_start + 5)
        pdf.cell(60, 6, labels[i], ln=2, align="C")
        pdf.set_font("Arial", "B", 16)
        pdf.cell(60, 10, str(values[i]), ln=1, align="C")

    pdf.ln(35)

    # Breakdown Stats
    pdf.set_font("Arial", "B", 14)
    pdf.set_text_color(255, 255, 255)
    pdf.cell(0, 10, "Condition Breakdown", ln=True)
    pdf.set_font("Arial", "", 12)
    pdf.set_text_color(220, 220, 220)
    pdf.cell(0, 8, f"- Normal Panels: {total_normal}", ln=True)
    pdf.cell(0, 8, f"- Dusty Panels: {total_dusty}", ln=True)
    pdf.cell(0, 8, f"- Cracked Panels: {total_cracked}", ln=True)

    # Key Metrics with percentages
    pdf.ln(5)
    pdf.set_font("Arial", "B", 14)
    pdf.set_text_color(255, 255, 255)
    pdf.cell(0, 10, "Key Metrics", ln=True)

    pdf.set_font("Arial", "", 12)
    pdf.set_text_color(240, 240, 240)
    if total_panels > 0:
        pct_normal = f"{(total_normal / total_panels) * 100:.1f}%"
        pct_dusty = f"{(total_dusty / total_panels) * 100:.1f}%"
        pct_cracked = f"{(total_cracked / total_panels) * 100:.1f}%"
    else:
        pct_normal = pct_dusty = pct_cracked = "0.0%"

    pdf.cell(0, 8, f"- Normal Panels: {total_normal} panels ({pct_normal})", ln=True)
    pdf.cell(0, 8, f"- Dusty Panels: {total_dusty} panels ({pct_dusty})", ln=True)
    pdf.cell(0, 8, f"- Cracked Panels: {total_cracked} panels ({pct_cracked})", ln=True)

    # Table Header
    pdf.ln(8)
    pdf.set_fill_color(30, 41, 59)
    pdf.set_text_color(255, 255, 255)
    pdf.set_font("Arial", "B", 12)
    pdf.cell(70, 10, "File", 1, 0, 'C', fill=True)
    pdf.cell(30, 10, "Total", 1, 0, 'C', fill=True)
    pdf.cell(30, 10, "Normal", 1, 0, 'C', fill=True)
    pdf.cell(30, 10, "Dusty", 1, 0, 'C', fill=True)
    pdf.cell(30, 10, "Cracked", 1, 1, 'C', fill=True)

    # Table Rows
    pdf.set_font("Arial", "", 12)
    pdf.set_text_color(255, 255, 255)
    for idx, row in summary_df.iterrows():
        pdf.cell(70, 8, str(row["File"]), 1)
        pdf.cell(30, 8, str(row["Total Panels"]), 1, 0, 'C')
        pdf.cell(30, 8, str(row["Normal"]), 1, 0, 'C')
        pdf.cell(30, 8, str(row["Dusty"]), 1, 0, 'C')
        pdf.cell(30, 8, str(row["Cracked"]), 1, 1, 'C')

    # Export PDF
    buffer = BytesIO()
    pdf_bytes = pdf.output(dest='S').encode('latin1')
    buffer.write(pdf_bytes)
    buffer.seek(0)
    buffer.name = "AeroAI_Dashboard_Report.pdf"
    return buffer