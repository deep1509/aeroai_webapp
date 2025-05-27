import streamlit as st
import pandas as pd
from datetime import datetime
from pathlib import Path

# Mock apply_dark_theme for standalone execution if theme.py is not provided
def apply_dark_theme():
    st.markdown(
        """
        <style>
        html, body, [data-testid="stAppViewContainer"] {
            background-color: #0F172A !important; /* Dark background */
            color: #E2E8F0 !important; /* Light text color */
        }
        [data-testid="stSidebar"] {
            background-color: #1E293B !important; /* Slightly lighter dark for sidebar */
            color: #E2E8F0 !important;
        }
        /* Ensure headers and other elements also use light text */
        h1, h2, h3, h4, h5, h6, p, li, div, span {
            color: #E2E8F0 !important;
        }
        /* Specific overrides for elements that might not inherit */
        .stMarkdown, .stText, .stAlert, .stInfo, .stWarning, .stSuccess {
            color: #E2E8F0 !important;
        }
        /* Adjust Streamlit specific components for dark theme */
        .stButton>button {
            background-color: #2D3748;
            color: #E2E8F0;
            border: none;
            border-radius: 8px;
            padding: 10px 20px;
            cursor: pointer;
        }
        .stButton>button:hover {
            background-color: #4A5568;
        }
        .stTextInput>div>div>input {
            background-color: #1E293B;
            color: #E2E8F0;
            border: 1px solid #4A5568;
            border-radius: 8px;
        }
        .stSelectbox>div>div {
            background-color: #1E293B;
            color: #E2E8F0;
            border: 1px solid #4A5568;
            border-radius: 8px;
        }
        .stSelectbox>div>div>div[data-baseweb="select"] {
            color: #E2E8F0;
        }
        /* Ensure the dropdown options are also visible */
        .stSelectbox div[role="listbox"] {
            background-color: #1E293B !important;
            color: #E2E8F0 !important;
        }
        .stSelectbox div[role="option"] {
            color: #E2E8F0 !important;
        }
        
        /* NEW: More specific rules for selectbox selected value and dropdown options */
        /* For the displayed selected value within the selectbox */
        .stSelectbox div[data-baseweb="select"] > div > div {
            color: #E2E8F0 !important;
        }

        /* For the text of individual options within the dropdown list */
        .stSelectbox div[role="listbox"] div[role="option"] {
            color: #E2E8F0 !important;
        }

        </style>
        """,
        unsafe_allow_html=True
    )