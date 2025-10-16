import tensorflow as tf
import numpy as np
import joblib

# Chargement du modèle et du scaler
model = tf.keras.models.load_model("models/ecg_best_model.h5", compile=False)
scaler = joblib.load("models/scaler_ecg.pkl")

label_map = {
    0: "Normal",
    1: "Supraventriculaire",
    2: "Extrasystole ventriculaire",
    3: "Fusion de battements",
    4: "Bruit / Autre"
}

def predict_ecg(signal_1d):
    try:
        x = signal_1d.reshape(1, -1)
        x = x[..., np.newaxis]  # (1, 187, 1)
        y_pred = model.predict(x, verbose=0)
        label = int(np.argmax(y_pred))
        confidence = float(np.max(y_pred))
        return label_map[label], confidence, {label_map[i]: float(y_pred[0][i]) for i in range(5)}
    except Exception as e:
        print(f"Erreur ECG : {e}")
        return None, 0.0, {}

