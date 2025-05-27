# home_page.py
import streamlit as st
import base64
from pathlib import Path

def get_image_as_base64(path):
    """Encodes an image file to a Base64 string."""
    try:
        with open(path, "rb") as image_file:
            encoded_string = base64.b64encode(image_file.read()).decode()
        return f"data:image/png;base64,{encoded_string}" # Adjust mime type if not PNG
    except FileNotFoundError:
        st.error(f"Error: Image not found at {path}. Make sure the 'assets' folder is in the correct directory.")
        return ""

def show_home_page():
    # Path to your logo file
    logo_path = Path("assets/logo2.png")
    logo_base64 = get_image_as_base64(logo_path)

    # Inject dark theme background with custom font & styles
    st.markdown("""
        <style>
            body, [data-testid="stAppViewContainer"] {
                background-color: #0F172A;
            }
            .hero {
                background-image: url('https://images.unsplash.com/photo-1615205242059-8d4e5848741e?auto=format&fit=crop&w=1470&q=80');
                background-size: cover;
                background-position: center;
                padding: 4rem;
                border-radius: 12px;
            }
            .hero-title {
                font-size: 3.5rem;
                font-weight: 800;
                color: #22D3EE;
                text-align: center;
                margin-bottom: 1rem;
            }
            .hero-subtitle {
                font-size: 1.4rem;
                color: #CBD5E1;
                text-align: center;
                margin-bottom: 2rem;
            }
            /* Remove the direct styling for .cta-btn button as we'll use Streamlit's button */
            /* We'll use inline CSS for the st.button to apply these styles */
        </style>
    """, unsafe_allow_html=True)

    # Layout wrapper - we will inject the Streamlit button here
    st.markdown(f"""
        <div class="hero">
            <div style="text-align: center; margin-bottom: 2rem;">
                <img src="{logo_base64}" width="180" />
            </div>
            <div class="hero-title">Welcome to AeroAI</div>
            <div class="hero-subtitle">AI-powered Drone Platform for Real-Time Solar Panel Inspection</div>
            <div class="hero-subtitle">Detect anomalies, assess performance, and optimize your solar operations using state-of-the-art computer vision models.</div>
            <div style='text-align: center; margin-top: 3rem; color: #94A3B8;'>
                Group 04<br>
                Team Members: Shruti Agarwal, Deep Patel, Kush Patel
            </div>
        </div>
    """, unsafe_allow_html=True)

    # Use a Streamlit button and apply the custom styles via its parameters
    # The button needs to be placed *outside* the st.markdown block to be interactive
    col1, col2, col3 = st.columns([1, 0.5, 1]) # Use columns to center the button
    with col2: # Place button in the middle column
        if st.button("🚀 Get Started", key="get_started_btn"):
            st.session_state.page = "📤 Upload Media"

    # Apply custom styling to the Streamlit button
    st.markdown("""
        <style>
            /* Target the specific Streamlit button based on its data-testid */
            button[data-testid="stButton"] {
                background-color: #6366F1;
                color: white;
                font-size: 1.1rem;
                padding: 0.75rem 2rem;
                border: none;
                border-radius: 8px;
                font-weight: 600;
                transition: 0.3s ease;
                width: 100%; /* Make button fill its column */
            }
            button[data-testid="stButton"]:hover {
                filter: brightness(1.1);
                cursor: pointer;
            }
        </style>
    """, unsafe_allow_html=True)

# Example of how to call it if this were your main app.py
# if __name__ == "__main__":
#     if 'page' not in st.session_state:
#         st.session_state.page = "🏠 Home" # Initialize default page

#     if st.session_state.page == "🏠 Home":
#         show_home_page()
#     elif st.session_state.page == "📤 Upload Media":
#         st.write("Navigated to Upload Media page!") # Replace with your upload page function