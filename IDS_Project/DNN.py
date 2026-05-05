import pandas as pd
import numpy as np
import time
import os
import warnings
warnings.filterwarnings('ignore')

import joblib
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import (accuracy_score, f1_score, precision_score,
                             recall_score, roc_auc_score, confusion_matrix)
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import roc_curve

import tensorflow as tf
from tensorflow.keras.models import Sequential, load_model
from tensorflow.keras.layers import Dense, Dropout, BatchNormalization
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau

DATA_PATH   = r"C:\Users\user\Desktop\cicids_clean.csv"
SCALER_PATH = r"C:\Users\user\Desktop\rf_scaler.pkl"
DNN_PATH    = r"C:\Users\user\Desktop\dnn_model.keras"
print("=" * 60)
print("DNN — IDS")
print("=" * 60)
# --- 1. Деректер ---
print("\n[1] Деректер жүктелуде...")
chunks = []
for chunk in pd.read_csv(DATA_PATH, low_memory=False, chunksize=100_000):
    chunks.append(chunk)
    if sum(len(c) for c in chunks) >= 500_000:
        break
df = pd.concat(chunks, ignore_index=True)
print(f"Жолдар: {len(df):,} | Бағандар: {len(df.columns)}")
# --- 2. Дайындау ---
print("\n[2] Деректерді дайындау...")
y = df['binary_label'].values
X = df.drop(columns=['binary_label']).values
if os.path.exists(SCALER_PATH):
    scaler = joblib.load(SCALER_PATH)
    X = scaler.transform(X)
else:
    scaler = MinMaxScaler()
    X = scaler.fit_transform(X)
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)
print(f"Train: {len(X_train):,} | Test: {len(X_test):,}")
callbacks = [
    EarlyStopping(monitor='val_loss', patience=5, restore_best_weights=True),
    ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=3)
]
# --- 3. DNN ---
print("\n" + "=" * 60)
print("DNN")
print("=" * 60)
if os.path.exists(DNN_PATH):
    print("Модель жүктелді...")
    dnn_model = load_model(DNN_PATH)
    train_time_dnn = 0.0
else:
    print("Модель оқытылуда...")
    n_features = X_train.shape[1]
    dnn_model = Sequential([
        Dense(256, activation='relu', input_shape=(n_features,)),
        BatchNormalization(),
        Dropout(0.3),
        Dense(128, activation='relu'),
        BatchNormalization(),
        Dropout(0.3),
        Dense(64, activation='relu'),
        Dropout(0.2),
        Dense(1, activation='sigmoid')
    ])
    dnn_model.compile(
        optimizer='adam',
        loss='binary_crossentropy',
        metrics=['accuracy']
    )
    t = time.time()
    dnn_model.fit(
        X_train, y_train,
        epochs=50,
        batch_size=1024,
        validation_split=0.1,
        callbacks=callbacks,
        verbose=1
    )
    train_time_dnn = time.time() - t
    dnn_model.save(DNN_PATH)
    print(f"Модель сақталды: {DNN_PATH}")
# --- 4. Тест ---
print("\n[3] Тестілеу...")
t = time.time()
y_prob = dnn_model.predict(X_test, verbose=0).flatten()
y_pred = (y_prob > 0.5).astype(int)
infer_time = (time.time() - t) * 1000
acc = accuracy_score(y_test, y_pred)
f1 = f1_score(y_test, y_pred)
pre = precision_score(y_test, y_pred)
rec = recall_score(y_test, y_pred)
auc = roc_auc_score(y_test, y_prob)
print("\n" + "=" * 60)
print("НӘТИЖЕ")
print("=" * 60)
print(f"Accuracy:  {acc*100:.2f}%")
print(f"F1:        {f1*100:.2f}%")
print(f"Precision: {pre*100:.2f}%")
print(f"Recall:    {rec*100:.2f}%")
print(f"ROC-AUC:   {auc*100:.2f}%")
print(f"Train time: {train_time_dnn:.1f} sec")
print(f"Inference:  {infer_time/len(X_test):.4f} ms")
# --- 5. Графиктер ---
print("\n[4] Графиктер...")
fig, axes = plt.subplots(1, 2, figsize=(12, 5))
cm = confusion_matrix(y_test, y_pred)
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', ax=axes[0])
axes[0].set_title("Confusion Matrix")
fpr, tpr, _ = roc_curve(y_test, y_prob)
axes[1].plot(fpr, tpr)
axes[1].plot([0, 1], [0, 1], 'k--')
axes[1].set_title("ROC Curve")
plt.tight_layout()
save_path = r"C:\Users\user\Desktop\dnn_results.png"
plt.savefig(save_path, dpi=150)
print(f"Сақталды: {save_path}")