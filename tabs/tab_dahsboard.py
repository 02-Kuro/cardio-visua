# tabs/tab_dashboard.py
import streamlit as st
from tabs.tab_mri import get_segmentation_path_from_image
from utils.visualize_3d import show_heart_3d
from utils.predict_ecg import predict_ecg
import pandas as pd
import numpy as np
import os

def tab_dashboard():
    st.markdown("<h2 style='color: red;'>🩺 Tableau de bord Cardio-Visua</h2>", unsafe_allow_html=True)
    st.markdown("<h4 style='color: white;'>Glissez un fichier IRM (.nii.gz) ou ECG (.csv) pour afficher automatiquement la prédiction et la visualisation 3D.</h4>", unsafe_allow_html=True)

    col1, col2 = st.columns([2, 1])  # 2/3 pour IRM, 1/3 pour ECG

    # === Bloc IRM
    with col1:
        st.markdown("### 📂 IRM Cardiaque")
        uploaded_irm = st.file_uploader("Glisser un fichier .nii.gz", type=["nii.gz"], key="irm_uploader")

        if uploaded_irm:
            temp_irm_path = os.path.join("app", "temp", uploaded_irm.name)
            with open(temp_irm_path, "wb") as f:
                f.write(uploaded_irm.getbuffer())

            seg_path = get_segmentation_path_from_image(temp_irm_path)

            if os.path.exists(seg_path):
                st.success("✅ IRM chargée. Visualisation ci-dessous ⬇️")
                show_heart_3d(temp_irm_path, seg_path)
            else:
                st.error("❌ Segmentation correspondante introuvable.")

    # === Bloc ECG
    with col2:
        st.markdown("### 💓 Analyse ECG")
        uploaded_ecg = st.file_uploader("Déposer un fichier ECG (.csv)", type="csv", key="ecg_uploader")

        if uploaded_ecg:
            try:
                df = pd.read_csv(uploaded_ecg, header=None)
                signal = df.iloc[0, :-1].values.astype(float)

                if signal.shape[0] != 187:
                    st.error(f"⚠️ Le signal doit avoir 187 points (actuellement {signal.shape[0]}).")
                    return

                # Affichage signal
                st.line_chart(signal)

                # Prédiction ECG
                label, confidence, probas = predict_ecg(signal)

                st.markdown(f"### 🧠 Classe prédite : `{label}`")
                st.markdown(f"**Confiance :** `{confidence*100:.2f}%`")

                st.subheader("🔢 Probabilités")
                proba_df = pd.DataFrame.from_dict(probas, orient='index', columns=['Probabilité (%)'])
                proba_df['Probabilité (%)'] *= 100
                st.dataframe(proba_df.sort_values(by='Probabilité (%)', ascending=False).style.format({"Probabilité (%)": "{:.2f}"}))

            except Exception as e:
                st.error(f"❌ Erreur ECG : {e}")
