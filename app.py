import streamlit as st
import pandas as pd
import numpy as np
import os
import cv2
from PIL import Image
from scipy.stats import skew, kurtosis, entropy

st.set_page_config(page_title="Forgery Dataset Feature Extractor", layout="wide")
st.title("✍ Forged Handwritten Document Database - Auto Class Detection & Feature Extraction")

def extract_features(image_path, class_label):
    try:
        img = cv2.imread(image_path)
        if img is None:
            return {"file_name": os.path.basename(image_path), "class": class_label, "error": "Unreadable file"}
        
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        height, width = gray.shape
        file_size = os.path.getsize(image_path) / 1024  # KB
        aspect_ratio = round(width / height, 3)

        mean_intensity = np.mean(gray)
        std_intensity = np.std(gray)
        skewness = skew(gray.flatten())
        kurt = kurtosis(gray.flatten())

        hist = np.histogram(gray, bins=256, range=(0, 255))[0]
        shannon_entropy = entropy(hist + 1e-9)

        edges = cv2.Canny(gray, 100, 200)
        edge_density = np.mean(edges > 0)

        return {
            "file_name": os.path.basename(image_path),
            "class": class_label,
            "width": width,
            "height": height,
            "aspect_ratio": aspect_ratio,
            "file_size_kb": round(file_size, 2),
            "mean_intensity": round(mean_intensity, 3),
            "std_intensity": round(std_intensity, 3),
            "skewness": round(skewness, 3),
            "kurtosis": round(kurt, 3),
            "entropy": round(shannon_entropy, 3),
            "edge_density": round(edge_density, 3)
        }
    except Exception as e:
        return {"file_name": image_path, "class": class_label, "error": str(e)}

dataset_root = st.text_input("📂 Enter dataset root path:", "")

if dataset_root and os.path.isdir(dataset_root):
    st.info("🔎 Scanning dataset...")
    records = []

    classes = [f for f in os.listdir(dataset_root) if os.path.isdir(os.path.join(dataset_root, f))]
    st.success(f" Detected {len(classes)} classes: {classes}")

    for class_dir in classes:
        class_path = os.path.join(dataset_root, class_dir)
        files = [f for f in os.listdir(class_path) if f.lower().endswith((".jpg", ".jpeg", ".png"))]

        st.write(f" Class '{class_dir}' → {len(files)} images")
        for fname in files:
            path = os.path.join(class_path, fname)
            rec = extract_features(path, class_dir)
            records.append(rec)

    df = pd.DataFrame(records)
    st.subheader("Features Extracted (Preview)")
    st.dataframe(df)  # Full preview with scroll

    save_path = os.path.join(dataset_root, "metadata_features.csv")
    df.to_csv(save_path, index=False)
    st.success(f"Features saved to {save_path}")

    if "class" in df.columns:
        st.subheader("Class Distribution")
        st.bar_chart(df["class"].value_counts())

    st.subheader("🖼 Sample Images")
    cols = st.columns(5)
    for idx, cls in enumerate(classes):
        class_samples = [f for f in os.listdir(os.path.join(dataset_root, cls)) if f.lower().endswith((".jpg", ".jpeg", ".png"))]
        if class_samples:
            sample_img = os.path.join(dataset_root, cls, class_samples[0])
            img = Image.open(sample_img)
            cols[idx % 5].image(img, caption=cls, width='stretch')

elif dataset_root:
    st.error("Invalid dataset path. Please enter a valid folder.")
