import os
import pickle
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
import pandas as pd

# ---- Paths ----
ART_DIR = "d:/Digital Forensics Scanner/proceed_data"
RESULTS_DIR = "d:/Digital Forensics Scanner/results"
MODEL_PATH = os.path.join(ART_DIR, "scanner_hybrid_final.keras")
ENCODER_PATH = os.path.join(ART_DIR, "hybrid_label_encoder.pkl")
TEST_DATA_PATH = os.path.join(ART_DIR, "hybrid_test_data.pkl")
REPORT_PATH = os.path.join(RESULTS_DIR, "classification_report.csv")
CONF_MATRIX_PATH = os.path.join(RESULTS_DIR, "CNN_confusion_matrix.png")

# Create results directory if it doesn't exist
os.makedirs(RESULTS_DIR, exist_ok=True)

# ---- Load label encoder and test data ----
with open(ENCODER_PATH, "rb") as f:
    le = pickle.load(f)

with open(TEST_DATA_PATH, "rb") as f:
    test_data = pickle.load(f)
    X_img_te = test_data["X_img_te"]
    X_feat_te = test_data["X_feat_te"]
    y_te = test_data["y_te"]

# Convert one-hot encoded y_te back to integer labels for evaluation
y_int_te = np.argmax(y_te, axis=1)

# ---- Load model ----
model = tf.keras.models.load_model(MODEL_PATH)

# ---- Evaluate ----
y_pred_prob = model.predict([X_img_te, X_feat_te])
y_pred = np.argmax(y_pred_prob, axis=1)

test_acc = accuracy_score(y_int_te, y_pred)
print(f"Test Accuracy: {test_acc*100:.2f}%")

# ---- Classification Report ----
print("\nClassification Report:")
report = classification_report(y_int_te, y_pred, target_names=le.classes_, output_dict=True)
print(classification_report(y_int_te, y_pred, target_names=le.classes_))

df_report = pd.DataFrame(report).transpose()
df_report.to_csv(REPORT_PATH)
print(f"Classification report saved to: {REPORT_PATH}")

# ---- Confusion Matrix ----
cm = confusion_matrix(y_int_te, y_pred)
plt.figure(figsize=(10, 8))
sns.heatmap(cm, annot=True, fmt="d", cmap="Blues",
            xticklabels=le.classes_, yticklabels=le.classes_)
plt.xlabel("Predicted")
plt.ylabel("True")
plt.title("Confusion Matrix")
plt.xticks(rotation=45)
plt.yticks(rotation=0)
plt.savefig(CONF_MATRIX_PATH, dpi=300, bbox_inches="tight")
print(f"Confusion matrix saved to: {CONF_MATRIX_PATH}")
plt.close()