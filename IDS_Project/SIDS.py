import pandas as pd
import numpy as np
import time
import os
import warnings
warnings.filterwarnings('ignore')
from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    accuracy_score, f1_score, precision_score,
    recall_score, roc_auc_score, confusion_matrix
)
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import roc_curve
DATA_PATH = r"C:\Users\user\Desktop\cicids_clean.csv"
print("=" * 60)
print("SIDS — ШЕКТІ ӘДІС (ML-сіз)")
print("=" * 60)
# --- 1. Деректер ---
print("\n[1] Деректер жүктелуде...")
chunks = []
for chunk in pd.read_csv(DATA_PATH, chunksize=100_000, low_memory=False):
    chunks.append(chunk)
    if sum(len(c) for c in chunks) >= 500_000:
        break
df = pd.concat(chunks, ignore_index=True)
print(f"Жолдар: {len(df):,} | Бағандар: {len(df.columns)}")
# --- 2. Дайындау ---
print("\n[2] Дайындау...")
y = df['binary_label'].values
X = df.drop(columns=['binary_label'])
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)
print(f"Test: {len(X_test):,}")
# --- 3. SIDS ережелері ---
print("\n[3] Ережелік модель...")
def sids_predict(df_input):
    score = np.zeros(len(df_input))
    if 'Flow Packets/s' in df_input.columns:
        col = pd.to_numeric(df_input['Flow Packets/s'], errors='coerce').fillna(0)
        score += (col > 1000) * 2
    if 'Flow Bytes/s' in df_input.columns:
        col = pd.to_numeric(df_input['Flow Bytes/s'], errors='coerce').fillna(0)
        score += (col > 1_000_000) * 2
    if 'Average Packet Size' in df_input.columns:
        col = pd.to_numeric(df_input['Average Packet Size'], errors='coerce').fillna(0)
        score += (col < 10) * 1
    if 'Flow Duration' in df_input.columns:
        col = pd.to_numeric(df_input['Flow Duration'], errors='coerce').fillna(0)
        score += (col < 100) * 1
    if 'SYN Flag Count' in df_input.columns:
        col = pd.to_numeric(df_input['SYN Flag Count'], errors='coerce').fillna(0)
        score += (col > 5) * 2
    if 'Init_Win_bytes_forward' in df_input.columns:
        col = pd.to_numeric(df_input['Init_Win_bytes_forward'], errors='coerce').fillna(0)
        score += (col == 0) * 1
    return (score >= 2).astype(int)
# --- 4. Болжам ---
t = time.time()
y_pred = sids_predict(X_test)
infer_time = (time.time() - t) * 1000
# ықтималдық (ROC үшін)
def sids_score(df_input):
    score = np.zeros(len(df_input))
    if 'Flow Packets/s' in df_input.columns:
        col = pd.to_numeric(df_input['Flow Packets/s'], errors='coerce').fillna(0)
        score += (col > 1000) * 2
    if 'Flow Bytes/s' in df_input.columns:
        col = pd.to_numeric(df_input['Flow Bytes/s'], errors='coerce').fillna(0)
        score += (col > 1_000_000) * 2
    if 'Average Packet Size' in df_input.columns:
        col = pd.to_numeric(df_input['Average Packet Size'], errors='coerce').fillna(0)
        score += (col < 10) * 1
    if 'Flow Duration' in df_input.columns:
        col = pd.to_numeric(df_input['Flow Duration'], errors='coerce').fillna(0)
        score += (col < 100) * 1
    if 'SYN Flag Count' in df_input.columns:
        col = pd.to_numeric(df_input['SYN Flag Count'], errors='coerce').fillna(0)
        score += (col > 5) * 2
    if 'Init_Win_bytes_forward' in df_input.columns:
        col = pd.to_numeric(df_input['Init_Win_bytes_forward'], errors='coerce').fillna(0)
        score += (col == 0) * 1
    return score / (score.max() if score.max() > 0 else 1)
y_prob = sids_score(X_test)
# --- 5. Метрикалар ---
acc = accuracy_score(y_test, y_pred)
f1 = f1_score(y_test, y_pred)
pre = precision_score(y_test, y_pred, zero_division=0)
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
print(f"Inference: {infer_time/len(X_test):.4f} ms")
# --- 6. Графиктер ---
print("\n[4] Графиктер...")
cm = confusion_matrix(y_test, y_pred)
fig, axes = plt.subplots(1, 2, figsize=(12, 5))
sns.heatmap(cm, annot=True, fmt='d', cmap='Reds', ax=axes[0])
axes[0].set_title("Confusion Matrix")
fpr, tpr, _ = roc_curve(y_test, y_prob)
axes[1].plot(fpr, tpr, color='red')
axes[1].plot([0, 1], [0, 1], 'k--')
axes[1].set_title("ROC Curve")
plt.tight_layout()
save_path = r"C:\Users\user\Desktop\sids_results.png"
plt.savefig(save_path, dpi=150)
print(f"Сақталды: {save_path}")