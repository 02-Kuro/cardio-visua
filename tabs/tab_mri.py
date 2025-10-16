# tab_mri.py
import os
import streamlit as st
from utils.visualize_3d import show_heart_3d

def get_segmentation_path_from_image(img_path):
    base_name = os.path.basename(img_path).replace(".nii.gz", "")
    seg_name = base_name + "_pred.nii.gz"
    seg_path = os.path.join("data", "prediction", seg_name)
    return seg_path

def tab_mri():
    st.subheader("🧠 IRM Cardiaque – Prédiction et visualisation 3D")

    uploaded_file = st.file_uploader("📂 Glissez une IRM (.nii.gz)", type="nii.gz")

    if uploaded_file is not None:
        # Sauvegarde temporaire
        temp_path = os.path.join("app", "data", "temp", uploaded_file.name)
        os.makedirs(os.path.dirname(temp_path), exist_ok=True)
        with open(temp_path, "wb") as f:
            f.write(uploaded_file.getbuffer())
        
        st.success(f"✅ Fichier chargé : {uploaded_file.name}")

        # Construction du chemin segmentation
        seg_path = get_segmentation_path_from_image(temp_path)

        # Vérification que la segmentation existe
        if os.path.exists(seg_path):
            show_heart_3d(temp_path, seg_path)
        else:
            st.error("❌ Segmentation correspondante introuvable. Assurez-vous que le fichier *_pred.nii.gz existe dans app/data/prediction/")
