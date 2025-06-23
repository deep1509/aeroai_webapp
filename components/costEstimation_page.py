import streamlit as st
import pandas as pd
from datetime import datetime
from pathlib import Path
from fpdf import FPDF
from io import BytesIO
from datetime import datetime

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

    cleaning_only_cost = total_dusty * COST_CLEANING
    extra_cost = total_cost - cleaning_only_cost  # ✅ Add this line

    pdf_data = generate_cost_report_pdf(
    df, panel_type, total_cost, total_dusty, total_cracked,
    most_expensive, extra_cost, cleaning_only_cost, COST_CLEANING, COST_REPLACEMENT
    )



    st.download_button(
        label="📄 Download PDF Report",
        data=pdf_data,
        file_name="AeroAI_Cost_Report.pdf",
        mime="application/pdf"
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


# 🚀 Upgraded PDF Report for AeroAI (TechFest-Ready)
from fpdf import FPDF
from datetime import datetime
from io import BytesIO
import matplotlib.pyplot as plt
import tempfile

class AeroPDF(FPDF):
    def header(self):
        self.set_fill_color(15, 23, 42)  # Dark navy
        self.rect(0, 0, 210, 25, 'F')
        self.image("assets/logo1.png", 10, 5, 15)
        self.set_font("Arial", "B", 14)
        self.set_text_color(255, 255, 255)
        self.set_y(8)
        self.cell(0, 10, "AeroAI - Solar Panel Inspection Report", ln=True, align="C")
        self.ln(5)

    def footer(self):
        self.set_y(-15)
        self.set_font("Arial", "I", 8)
        self.set_text_color(180, 180, 180)
        self.cell(0, 10, f"Generated on {datetime.now().strftime('%Y-%m-%d %H:%M:%S')} | Page {self.page_no()}", align="C")

def generate_cost_report_pdf(df, panel_type, total_cost, total_dusty, total_cracked, most_expensive, extra_cost, cleaning_only_cost, cleaning_cost, replacement_cost):
    pdf = AeroPDF()
    pdf.add_page()

    # Background fill
    pdf.set_fill_color(15, 23, 42)
    pdf.rect(0, 25, 210, 275, 'F')

    pdf.set_font("Arial", "B", 16)
    pdf.set_text_color(255, 165, 0)
    pdf.ln(20)
    pdf.cell(0, 10, "Cost Estimation Summary", ln=True, align="C")

    # Summary Cards
    y_start = pdf.get_y() + 5
    x_positions = [15, 75, 135]
    labels = ["Total Panels", "Dusty Panels", "Cracked Panels"]
    values = [total_dusty + total_cracked, total_dusty, total_cracked]
    colors = [(236, 253, 245), (252, 243, 207), (254, 226, 226)]

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

    # Total Cost Section
    pdf.set_font("Arial", "B", 14)
    pdf.set_text_color(250, 204, 21)
    pdf.cell(0, 10, f"Total Maintenance Cost: ${total_cost:,.0f}", ln=True, align="C")

    pdf.set_font("Arial", "", 12)
    pdf.set_text_color(200, 200, 200)
    pdf.ln(5)
    pdf.cell(0, 8, f"Panel Type Selected: {panel_type}", ln=True)
    pdf.cell(0, 8, f"Cleaning per Dusty Panel: ${cleaning_cost}", ln=True)
    pdf.cell(0, 8, f"Replacement per Cracked Panel: ${replacement_cost}", ln=True)

    pdf.ln(8)
    pdf.set_font("Arial", "B", 13)
    pdf.set_text_color(255, 255, 255)
    pdf.cell(0, 10, "Key Insights", ln=True)

    pdf.set_font("Arial", "", 12)
    pdf.cell(0, 8, f"- Most Expensive File: {most_expensive['File']} (${int(most_expensive['Cost'])})", ln=True)
    pdf.cell(0, 8, f"- Estimated Cost Without Cracks: ${cleaning_only_cost:,.0f}", ln=True)
    pdf.cell(0, 8, f"- Extra Cost Due to Cracks: ${extra_cost:,.0f}", ln=True)

    # File Table Header
    pdf.ln(8)
    pdf.set_fill_color(30, 41, 59)
    pdf.set_text_color(255, 255, 255)
    pdf.set_font("Arial", "B", 12)
    pdf.cell(80, 10, "File Name", 1, 0, 'C', fill=True)
    pdf.cell(30, 10, "Dusty", 1, 0, 'C', fill=True)
    pdf.cell(30, 10, "Cracked", 1, 0, 'C', fill=True)
    pdf.cell(40, 10, "Cost", 1, 1, 'C', fill=True)

    # Table Data
    pdf.set_font("Arial", "", 12)
    for idx, row in df.iterrows():
        pdf.cell(80, 8, str(row["File"]), 1)
        pdf.cell(30, 8, str(row["Dusty"]), 1, 0, 'C')
        pdf.cell(30, 8, str(row["Cracked"]), 1, 0, 'C')
        pdf.cell(40, 8, f"${row['Cost']}", 1, 1, 'C')


    # Export PDF
    buffer = BytesIO()
    pdf_bytes = pdf.output(dest='S').encode('latin1')
    buffer.write(pdf_bytes)
    buffer.seek(0)
    buffer.name = "AeroAI_Cost_Report.pdf"
    return buffer
