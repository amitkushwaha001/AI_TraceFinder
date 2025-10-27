import os, pickle, numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

# ---- Paths ----
ART_DIR   = "D:\Digital Forensics Scanner/proceed_data"
MODEL_PATH = os.path.join(ART_DIR, "scanner_hybrid_final.keras")
ENCODER_PATH = os.path.join(ART_DIR, "hybrid_label_encoder.pkl")
TEST_DATA_PATH = os.path.join(ART_DIR, "hybrid_test_data.pkl")

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
print("\nClassification Report:")
print(classification_report(y_int_te, y_pred, target_names=le.classes_))

cm = confusion_matrix(y_int_te, y_pred)
plt.figure(figsize=(10, 8))
sns.heatmap(cm, annot=True, fmt="d", cmap="Blues",
            xticklabels=le.classes_, yticklabels=le.classes_)
plt.xlabel("Predicted")
plt.ylabel("True")
plt.title("Confusion Matrix")
plt.xticks(rotation=45)
plt.yticks(rotation=0)
# Create results directory if it doesn't exist
os.makedirs("results", exist_ok=True)
save_path = "results/CNN_confusion_matrix.png"
plt.savefig(save_path, dpi=300, bbox_inches="tight")
print(f"Confusion matrix saved to: {save_path}")
plt.close()