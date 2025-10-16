import streamlit as st
import pandas as pd
import numpy as np
from utils.predict_ecg import predict_ecg  # ta fonction déjà existante

def tab_ecg():
    st.header("📈 Prédiction ECG")

    uploaded_file = st.file_uploader("Glisser-déposer un signal ECG (fichier .csv)", type="csv")
    
    if uploaded_file is not None:
        try:
            df = pd.read_csv(uploaded_file, header=None)
            raw_values = df.iloc[0, :-1].values  # Supposons que le signal est sur une seule ligne

            # Conversion en float
            signal = np.array([float(v) for v in raw_values])

            if signal.shape[0] != 187:
                st.error(f"❌ Le signal doit contenir exactement 187 points. Actuellement : {signal.shape[0]}")
                return

            # Affichage du signal ECG
            st.subheader("📊 Signal ECG")
            st.line_chart(signal)

            # Prédiction
            label, confidence, probas = predict_ecg(signal)

            if not isinstance(probas, dict) or len(probas) == 0:
                st.error("❌ Erreur dans la prédiction ECG.")
                return

            st.markdown(f"### 🧠 Classe prédite : `{label}`")
            st.markdown(f"**Confiance :** {confidence*100:.2f}%")

            st.subheader("🔢 Probabilités")
            proba_df = pd.DataFrame.from_dict(probas, orient='index', columns=['Probabilité (%)'])
            proba_df['Probabilité (%)'] *= 100
            st.dataframe(proba_df.sort_values(by='Probabilité (%)', ascending=False).style.format({"Probabilité (%)": "{:.2f}"}))

        except Exception as e:
            st.error(f"❌ Erreur lors de l'analyse ECG : {e}")
