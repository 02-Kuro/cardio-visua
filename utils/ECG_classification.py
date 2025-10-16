#!/usr/bin/env python
# coding: utf-8

# ECG Classification Notebook for mitbih_train.csv (MIT-BIH Dataset)
# Project: Cardio-Visua

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import classification_report, confusion_matrix
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv1D, MaxPooling1D, Flatten, Dense, Dropout, BatchNormalization
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.callbacks import EarlyStopping
import tensorflow as tf

# =========================
# 0. Check GPU
# =========================
gpus = tf.config.list_physical_devices('GPU')
print("✅ GPU available" if gpus else "❌ No GPU detected")

# =========================
# 1. Load and preprocess data
# =========================
print("\n📥 Loading dataset...")
df = pd.read_csv("C:/Users/kuro/Desktop/pfe/data/ECG/mitbih_train.csv", header=None)

X = df.iloc[:, :-1].values  # ECG signal (187 features)
y = df.iloc[:, -1].astype(int).values  # Labels (0-4)

# Normalize signals
scaler = StandardScaler()
X = scaler.fit_transform(X)
X = X[..., np.newaxis]  # Reshape to (samples, 187, 1)

# One-hot encode labels
y_cat = to_categorical(y, num_classes=5)

# Train/validation split
X_train, X_val, y_train, y_val = train_test_split(X, y_cat, test_size=0.2, stratify=y, random_state=42)

# =========================
# 2. Define CNN model
# =========================
print("\n🧠 Building CNN model...")
model = Sequential([
    Conv1D(32, kernel_size=5, activation='relu', input_shape=(187, 1)),
    BatchNormalization(),
    MaxPooling1D(pool_size=2),
    Dropout(0.2),

    Conv1D(64, kernel_size=5, activation='relu'),
    BatchNormalization(),
    MaxPooling1D(pool_size=2),
    Dropout(0.3),

    Flatten(),
    Dense(128, activation='relu'),
    Dropout(0.4),
    Dense(5, activation='softmax')  # 5 classes
])

model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
model.summary()

# =========================
# 3. Train the model
# =========================
print("\n🚀 Training model on GPU..." if gpus else "🚀 Training model on CPU...")

early_stopping = EarlyStopping(patience=5, restore_best_weights=True)

history = model.fit(
    X_train, y_train,
    validation_data=(X_val, y_val),
    epochs=30,
    batch_size=128,
    callbacks=[early_stopping],
    verbose=1
)

# =========================
# 4. Evaluate the model
# =========================
print("\n📊 Evaluation on validation set...")

y_pred = model.predict(X_val)
y_pred_labels = np.argmax(y_pred, axis=1)
y_true_labels = np.argmax(y_val, axis=1)

# Classification metrics
print("\n📈 Classification Report:")
print(classification_report(y_true_labels, y_pred_labels))

# Confusion Matrix
cm = confusion_matrix(y_true_labels, y_pred_labels)
plt.figure(figsize=(8, 6))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=range(5), yticklabels=range(5))
plt.xlabel("Predicted")
plt.ylabel("True")
plt.title("Confusion Matrix - ECG Classification")
plt.show()

# =========================
# 5. Plot training curves
# =========================
plt.figure(figsize=(12, 4))

# Accuracy
plt.subplot(1, 2, 1)
plt.plot(history.history['accuracy'], label='Train Accuracy')
plt.plot(history.history['val_accuracy'], label='Val Accuracy')
plt.title('Training vs Validation Accuracy')
plt.xlabel('Epoch')
plt.ylabel('Accuracy')
plt.legend()

# Loss
plt.subplot(1, 2, 2)
plt.plot(history.history['loss'], label='Train Loss')
plt.plot(history.history['val_loss'], label='Val Loss')
plt.title('Training vs Validation Loss')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.legend()

plt.tight_layout()
plt.show()


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import classification_report, confusion_matrix
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv1D, MaxPooling1D, Flatten, Dense, Dropout, BatchNormalization
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.callbacks import EarlyStopping
import tensorflow as tf
# ============================
# 🔍 Test on mitbih_test.csv
# ============================

print("\n📥 Loading mitbih_test.csv...")
test_df = pd.read_csv("C:/Users/kuro/Desktop/pfe/data/ECG/mitbih_test.csv", header=None)
X_test = test_df.iloc[:, :-1].values
y_test = test_df.iloc[:, -1].astype(int).values

# ⚙️ Apply same normalization
X_test = scaler.transform(X_test)
X_test = X_test[..., np.newaxis]  # Shape: (samples, 187, 1)

# 🎯 One-hot encode labels
y_test_cat = to_categorical(y_test, num_classes=5)

# 🚀 Run inference
print("\n🚀 Evaluating on test set...")
test_loss, test_accuracy = model.evaluate(X_test, y_test_cat, verbose=1)
print(f"\n✅ Test Accuracy: {test_accuracy:.4f}")
print(f"✅ Test Loss: {test_loss:.4f}")

# 📊 Classification report
y_pred_test = model.predict(X_test)
y_pred_labels_test = np.argmax(y_pred_test, axis=1)
y_true_labels_test = y_test

print("\n📈 Test Classification Report:")
print(classification_report(y_true_labels_test, y_pred_labels_test))

# 🔀 Confusion Matrix
print("🔍 Confusion Matrix:")
print(confusion_matrix(y_true_labels_test, y_pred_labels_test))


import joblib

model.save("ecg_best_model.h5")
joblib.dump(scaler, "scaler_ecg.pkl")


import tensorflow as tf
import numpy as np
import joblib

model = tf.keras.models.load_model("ecg_best_model.h5")
scaler = joblib.load("scaler_ecg.pkl")

label_map = {
    0: "Normal",
    1: "Supraventriculaire",
    2: "Extrasystole ventriculaire",
    3: "Fusion de battements",
    4: "Bruit / Autre"
}

def predict_ecg(signal_1d):
    x = scaler.transform(signal_1d.reshape(1, -1))
    x = x[..., np.newaxis]  # (1, 187, 1)
    y_pred = model.predict(x)
    label = np.argmax(y_pred)
    confidence = float(np.max(y_pred))
    return label_map[label], confidence


sig = X_test[0].flatten()  # exemple
classe, score = predict_ecg(sig)
print(f"Classe : {classe} ({score*100:.2f}%)")

