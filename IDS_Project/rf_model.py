import pandas as pd
import numpy as np
import time
import os
import warnings
warnings.filterwarnings('ignore')
import joblib
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import (
    accuracy_score, f1_score, precision_score,
    recall_score, roc_auc_score, confusion_matrix
)
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import roc_curve

DATA_PATH   = r"C:\Users\user\Desktop\cicids_clean.csv"
MODEL_PATH  = r"C:\Users\user\Desktop\rf_model.pkl"
SCALER_PATH = r"C:\Users\user\Desktop\rf_scaler.pkl"
print("=" * 60)
print("RANDOM FOREST — IDS")
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
print("\n[2] Деректер дайындалып жатыр...")
y = df['binary_label'].values
X = df.drop(columns=['binary_label']).values
scaler = MinMaxScaler()
X = scaler.fit_transform(X)
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)
print(f"Train: {len(X_train):,} | Test: {len(X_test):,}")
# --- 3. Модель ---
if os.path.exists(MODEL_PATH) and os.path.exists(SCALER_PATH):
    print("\n[3] Модель жүктелуде...")
    rf_model = joblib.load(MODEL_PATH)
    scaler = joblib.load(SCALER_PATH)
else:
    print("\n[3] Random Forest оқытылуда...")
    rf_model = RandomForestClassifier(
        n_estimators=100,
        max_depth=20,
        class_weight='balanced',
        random_state=42,
        n_jobs=-1
    )
    t = time.time()
    rf_model.fit(X_train, y_train)
    train_time = time.time() - t
    joblib.dump(rf_model, MODEL_PATH)
    joblib.dump(scaler, SCALER_PATH)
    print(f"Оқыту уақыты: {train_time:.1f} сек")
# --- 4. Тест ---
print("\n[4] Тестілеу...")
t = time.time()
y_pred = rf_model.predict(X_test)
infer_time = (time.time() - t) * 1000
y_prob = rf_model.predict_proba(X_test)[:, 1]
accuracy  = accuracy_score(y_test, y_pred)
f1        = f1_score(y_test, y_pred)
precision = precision_score(y_test, y_pred)
recall    = recall_score(y_test, y_pred)
roc_auc   = roc_auc_score(y_test, y_prob)
print("\n" + "=" * 60)
print("НӘТИЖЕ")
print("=" * 60)
print(f"Accuracy:  {accuracy*100:.2f}%")
print(f"F1:        {f1*100:.2f}%")
print(f"Precision: {precision*100:.2f}%")
print(f"Recall:    {recall*100:.2f}%")
print(f"ROC-AUC:   {roc_auc*100:.2f}%")
print(f"Inference: {infer_time/len(X_test):.4f} ms")
# --- 5. Графиктер ---
print("\n[5] Графиктер...")
cm = confusion_matrix(y_test, y_pred)
fig, axes = plt.subplots(1, 2, figsize=(14, 5))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', ax=axes[0])
axes[0].set_title("Confusion Matrix")
fpr, tpr, _ = roc_curve(y_test, y_prob)
axes[1].plot(fpr, tpr)
axes[1].plot([0, 1], [0, 1], 'k--')
axes[1].set_title("ROC Curve")
plt.tight_layout()
save_path = r"C:\Users\user\Desktop\rf_results.png"
plt.savefig(save_path, dpi=150)
print(f"Сақталды: {save_path}")