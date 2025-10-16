import streamlit as st
from PIL import Image
import os
from tabs.tab_mri import tab_mri
from tabs.tab_ecg import tab_ecg
from tabs.tab_dahsboard import tab_dashboard
import base64

st.set_page_config(
    page_title="Cardio-Visua",
    layout="wide",
    page_icon="🫀"
)

# === Convertir l’image logo en base64 pour fond de bannière
logo_path = os.path.join("assets", "logo.jpg")
banner_bg = ""
if os.path.exists(logo_path):
    with open(logo_path, "rb") as img_file:
        banner_bg = base64.b64encode(img_file.read()).decode()

# === CSS Hero Banner avec texte blanc
st.markdown(f"""
    <style>
    .hero-banner {{
        position: relative;
        width: 100%;
        height: 350px;
        background-image: url("data:image/png;base64,{banner_bg}");
        background-size: cover;
        background-position: center;
        border-radius: 10px;
        box-shadow: 0 4px 20px rgba(0,0,0,0.3);
        margin-bottom: 2rem;
    }}

    .hero-banner::after {{
        content: "";
        position: absolute;
        top: 0; left: 0;
        width: 100%; height: 100%;
        background: rgba(0, 0, 0, 0.6);
        border-radius: 10px;
    }}

    .hero-text {{
        position: absolute;
        top: 50%;
        left: 50%;
        transform: translate(-50%, -50%);
        color: white;
        text-align: center;
        z-index: 1;
    }}

    .hero-text h1 {{
        font-size: 3em;
        margin: 0;
        color: red;
    }}

    .hero-text h3 {{
        font-weight: normal;
        font-size: 1.4em;
        color: white;
    }}

    h1, h2, h3, h4, .stText, .stMarkdown, .stButton > button {{
        color: white !important;
    }}

    .main {{
        background-color: #0d1117;
    }}
    </style>

    <div class="hero-banner">
        <div class="hero-text">
            <h1>🫀 Cardio-Visua</h1>
            <h3>Plateforme de prédiction & visualisation des pathologies cardiaques</h3>
        </div>
    </div>
""", unsafe_allow_html=True)

# === Dashboard
tab_dashboard()
