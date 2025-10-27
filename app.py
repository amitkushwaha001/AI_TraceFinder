'''
Main Streamlit application file for AI TraceFinder.
'''
import streamlit as st
import cv2
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
import numpy as np
import joblib
from skimage.filters import sobel
from scipy.stats import skew, kurtosis, entropy
import pandas as pd
from PIL import Image
import matplotlib.pyplot as plt
import random
import hashlib
from tabulate import tabulate
import pickle
from tqdm import tqdm
from skimage.feature import local_binary_pattern
from scipy import ndimage
from scipy.fft import fft2, fftshift
import pywt
from concurrent.futures import ThreadPoolExecutor, as_completed
import tensorflow as tf
tf.compat.v1.logging.set_verbosity(tf.compat.v1.logging.ERROR)
import csv
import glob
import subprocess
import re

def set_app_style():
    '''Applies custom CSS styles to the Streamlit application.'''
    st.markdown("""
    <style>
    .stButton>button {
        background-color: #4CAF50;
        color: white;
        font-size: 16px;
        padding: 10px 24px;
        border-radius: 8px;
        border: none;
        transition-duration: 0.4s;
    }
    .stButton>button:hover {
        background-color: #45a049;
        color: white;
    }
    </style>
    """, unsafe_allow_html=True)

def home():
    '''Renders the Home page of the application.'''
    st.title("AI TraceFinder")
    st.markdown("Welcome to **AI TraceFinder**, your all-in-one solution for digital scanner forensics. "
                "This application leverages the power of AI to provide a comprehensive suite of tools for scanner identification, "
                "data analysis, and model evaluation.")
    st.markdown("Use the navigation panel on the left to explore the different functionalities of the application.")


def prediction():
    '''Renders the Prediction page.'''
    st.title("Scanner Identification")
    st.write("Upload an image to identify the scanner model using the Hybrid CNN model.")
    uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "png", "tif"])

    if uploaded_file is not None:
        st.image(uploaded_file, caption='Uploaded Image.', use_container_width=True)
        if st.button("Predict Scanner Model"):
            with st.spinner('Analyzing image and predicting with Hybrid CNN...'):
                try:
                    temp_path = os.path.join("temp_image.jpg")
                    with open(temp_path, "wb") as f:
                        f.write(uploaded_file.getbuffer())
                    
                    pred, conf, prob, classes = predict_with_hybrid_model(temp_path)
                    
                    st.success(f"Predicted Scanner Model: **{pred}**")
                    
                    st.metric(label="Confidence", value=f"{conf:.2f}%")

                    st.write("**Prediction Probabilities:**")
                    
                    prob_df = pd.DataFrame({
                        'Scanner Model': classes,
                        'Probability': prob
                    })
                    prob_df['Probability'] = prob_df['Probability'] * 100
                    prob_df = prob_df.sort_values(by='Probability', ascending=False).reset_index(drop=True)

                    st.dataframe(prob_df.style.format({'Probability': '{:.2f}%'}).highlight_max(axis=0, subset=['Probability']))

                except Exception as e:
                    st.error(f"An error occurred: {e}")
                finally:
                    if os.path.exists(temp_path):
                        os.remove(temp_path)


def eda():
    '''Renders the Exploratory Data Analysis page.'''
    st.title("Exploratory Data Analysis")
    st.write("This section provides an overview of the dataset used for training the models.")

    OFFICIAL_DIR = "d:/Digital Forensics Scanner/proceed_data/official"
    WIKI_DIR = "d:/Digital Forensics Scanner/proceed_data/Wikipedia"
    FLATFIELD_DIR = "d:/Digital Forensics Scanner/proceed_data/Flatfield"
    
    dataset_options = {
        "Official": OFFICIAL_DIR,
        "Wikipedia": WIKI_DIR,
        "Flatfield": FLATFIELD_DIR
    }

    dataset_choice = st.selectbox("Choose a dataset to analyze", list(dataset_options.keys()))

    if st.button("Run EDA"):
        with st.spinner(f"Running EDA on {dataset_choice} dataset..."):
            try:
                output, figures = run_eda(dataset_options[dataset_choice])
                st.code(output)
                for fig in figures:
                    st.pyplot(fig)
            except Exception as e:
                st.error(f"An error occurred during EDA: {e}")

def feature_extraction():
    '''Renders the Feature Extraction page.'''
    st.title("Feature Extraction")
    st.write("Run the feature extraction pipelines.")

    if st.button("Preprocess Images and Compute Residuals"):
        with st.spinner("Preprocessing images and computing residuals..."):
            preprocess_and_compute_residuals()
        st.success("Image preprocessing and residual computation complete.")

    if st.button("Compute Scanner Fingerprints"):
        with st.spinner("Computing scanner fingerprints..."):
            compute_scanner_fingerprints()
        st.success("Scanner fingerprint computation complete.")

    if st.button("Extract PRNU Features"):
        with st.spinner("Extracting PRNU features..."):
            extract_prnu_features()
        st.success("PRNU feature extraction complete.")

    if st.button("Extract Enhanced Features"):
        with st.spinner("Extracting enhanced features..."):
            extract_enhanced_features_func()
        st.success("Enhanced feature extraction complete.")

def testing():
    '''Renders the Testing page.'''
    st.title("Model Testing")
    st.write("Select a folder to run batch prediction on all images.")

    folder_path = st.text_input("Enter the path to the folder containing images:", "d:/Digital Forensics Scanner/proceed_data/Test")

    if st.button("Run Batch Prediction"):
        if os.path.isdir(folder_path):
            with st.spinner(f"Running batch prediction on folder: {folder_path}"):
                try:
                    results = predict_folder(folder_path, output_csv="hybrid_folder_results.csv")
                    st.success("Batch prediction complete!")
                    st.dataframe(pd.DataFrame(results, columns=["Image", "Predicted Label", "Confidence (%)"]))
                except Exception as e:
                    st.error(f"An error occurred during batch prediction: {e}")
        else:
            st.error("The specified path is not a valid directory.")

def cnn_model():
    '''Renders the CNN Model page.'''
    st.title("CNN Model Management")
    st.write("Train and evaluate the hybrid CNN model.")

    if st.button("Train Hybrid CNN Model"):
        with st.spinner("Training Hybrid CNN model..."):
            try:
                python_executable = "d:\\Digital Forensics Scanner\\.venv\\Scripts\\python.exe"
                train_script_path = "d:\\Digital Forensics Scanner\\src\\cnn_model\\train_hybrid_cnn.py"
                
                st.write("Starting training process...")
                progress_bar = st.progress(0)
                status_text = st.empty()
                
                process = subprocess.Popen(
                    [python_executable, train_script_path],
                    stdout=subprocess.PIPE,
                    stderr=subprocess.STDOUT,
                    text=True,
                    bufsize=1,
                    universal_newlines=True
                )

                output_lines = []
                total_epochs = 50  # Default, will be updated

                for line in iter(process.stdout.readline, ''):
                    output_lines.append(line)
                    # Show the last few lines of output
                    status_text.code("".join(output_lines[-10:]))

                    match = re.search(r"Epoch (\\d+)/(\\d+)", line)
                    if match:
                        epoch = int(match.group(1))
                        total_epochs = int(match.group(2))
                        progress_bar.progress(epoch / total_epochs)
                
                process.wait()
                progress_bar.progress(1.0)

                if process.returncode == 0:
                    st.success("Model training complete!")
                    history_path = "d:/Digital Forensics Scanner/proceed_data/hybrid_training_history.pkl"
                    if os.path.exists(history_path):
                        with open(history_path, "rb") as f:
                            history = pickle.load(f)
                        st.write("Training History:")
                        df_history = pd.DataFrame(history)
                        st.line_chart(df_history)
                else:
                    st.error("Model training failed.")
                    st.code("".join(output_lines))

            except Exception as e:
                st.error(f"An error occurred during training: {e}")

    if st.button("Evaluate Hybrid CNN Model"):
        with st.spinner("Evaluating Hybrid CNN model..."):
            try:
                python_executable = "d:\\Digital Forensics Scanner\\.venv\\Scripts\\python.exe"
                eval_script_path = "d:\\Digital Forensics Scanner\\src\\cnn_model\\eval_hybrid_cnn.py"

                result = subprocess.run([python_executable, eval_script_path], capture_output=True, text=True)

                st.success("Model evaluation complete!")
                
                output = result.stdout
                st.write("--- Evaluation Results ---")

                accuracy_match = re.search(r"Test Accuracy: (\\d+\\.\\d+)%", output)
                if accuracy_match:
                    accuracy = float(accuracy_match.group(1))
                    st.metric("Test Accuracy", f"{accuracy:.2f}%")
                else:
                    st.warning("Could not parse test accuracy from the script output.")

                report_match = re.search(r"Classification Report:\\s*\\n(.*?)(?:\\n\\s*\\n|\\Z)", output, re.DOTALL)
                if report_match:
                    report_str = report_match.group(1).strip()
                    st.text("Classification Report:")
                    st.code(report_str)
                else:
                    st.warning("Could not parse classification report from the script output.")
                    st.text("Raw output:")
                    st.code(output)

                if result.stderr:
                    st.write("Errors:")
                    st.code(result.stderr)

                conf_matrix_path = "results/CNN_confusion_matrix.png"
                if os.path.exists(conf_matrix_path):
                    st.image(conf_matrix_path, caption='Confusion Matrix', use_container_width=True)
                else:
                    st.warning("Confusion matrix image not found.")
            except Exception as e:
                st.error(f"An error occurred during evaluation: {e}")


@st.cache_resource
def load_hybrid_model_artifacts():
    ART_DIR = "d:/Digital Forensics Scanner/proceed_data"
    CKPT_PATH = os.path.join(ART_DIR, "scanner_hybrid_final.keras")
    hyb_model = tf.keras.models.load_model(CKPT_PATH, compile=False)

    with open(os.path.join(ART_DIR, "hybrid_label_encoder.pkl"), "rb") as f:
        le_inf = pickle.load(f)

    with open(os.path.join(ART_DIR, "hybrid_feat_scaler.pkl"), "rb") as f:
        scaler_inf = pickle.load(f)
    
    FP_PATH = os.path.join(ART_DIR, "Flatfield/scanner_fingerprints.pkl")
    with open(FP_PATH, "rb") as f:
        scanner_fps_inf = pickle.load(f)

    ORDER_NPY = os.path.join(ART_DIR, "Flatfield/fp_keys.npy")
    fp_keys_inf = np.load(ORDER_NPY, allow_pickle=True).tolist()
    
    return hyb_model, le_inf, scaler_inf, scanner_fps_inf, fp_keys_inf

def predict_with_hybrid_model(image_path):
    hyb_model, le_inf, scaler_inf, scanner_fps_inf, fp_keys_inf = load_hybrid_model_artifacts()
    
    IMG_SIZE = (256, 256)

    def corr2d(a, b):
        a = a.astype(np.float32).ravel(); b = b.astype(np.float32).ravel()
        a -= a.mean(); b -= b.mean()
        d = np.linalg.norm(a) * np.linalg.norm(b)
        return float((a @ b) / d) if d != 0 else 0.0

    def fft_radial_energy(img, K=6):
        f = np.fft.fftshift(np.fft.fft2(img))
        mag = np.abs(f)
        h, w = mag.shape; cy, cx = h//2, w//2
        yy, xx = np.ogrid[:h, :w]
        r = np.sqrt((yy - cy)**2 + (xx - cx)**2)
        rmax = r.max() + 1e-6
        bins = np.linspace(0, rmax, K+1)
        return [float(mag[(r >= bins[i]) & (r < bins[i+1])].mean() if ((r >= bins[i]) & (r < bins[i+1])).any() else 0.0) for i in range(K)]

    def lbp_hist_safe(img, P=8, R=1.0):
        rng = float(np.ptp(img))
        g = np.zeros_like(img, dtype=np.float32) if rng < 1e-12 else (img - img.min()) / (rng + 1e-8)
        g8 = (g * 255.0).astype(np.uint8)
        codes = local_binary_pattern(g8, P=P, R=R, method="uniform")
        hist, _ = np.histogram(codes, bins=np.arange(P+3), density=True)
        return hist.astype(np.float32).tolist()

    def preprocess_residual_pywt(path):
        img = cv2.imread(path, cv2.IMREAD_UNCHANGED)
        if img is None:
            raise ValueError(f"Cannot read {path}")
        if img.ndim == 3:
            img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        img = cv2.resize(img, IMG_SIZE, interpolation=cv2.INTER_AREA)
        img = img.astype(np.float32) / 255.0
        cA, (cH, cV, cD) = pywt.dwt2(img, 'haar')
        cH.fill(0); cV.fill(0); cD.fill(0)
        den = pywt.idwt2((cA, (cH, cV, cD)), 'haar')
        return (img - den).astype(np.float32)

    def make_feats_from_res(res):
        v_corr = [corr2d(res, scanner_fps_inf[k]) for k in fp_keys_inf]
        v_fft  = fft_radial_energy(res)
        v_lbp  = lbp_hist_safe(res)
        v = np.array(v_corr + v_fft + v_lbp, dtype=np.float32).reshape(1, -1)
        return scaler_inf.transform(v)

    res = preprocess_residual_pywt(image_path)
    x_img = np.expand_dims(res, axis=(0,-1))
    x_feat = make_feats_from_res(res)
    
    prob = hyb_model.predict([x_img, x_feat], verbose=0)[0]
    idx = int(np.argmax(prob))
    label = le_inf.classes_[idx]
    conf = float(prob[idx] * 100)
    
    return label, conf, prob, le_inf.classes_

def run_eda(dataset_path):
    output = ""
    figures = []

    output += f"\n=== EDA for {dataset_path} ===\n"
    
    class_counts = {}
    total_images = 0
    corrupted_files = []
    image_shapes = []
    brightness_values = []
    duplicates = []

    hashes = {}

    # Walk through dataset
    for class_name in os.listdir(dataset_path):
        class_path = os.path.join(dataset_path, class_name)
        if not os.path.isdir(class_path):
            continue

        subfolder_counts = {}
        class_total = 0

        for root, dirs, files in os.walk(class_path):
            sub_count = 0
            for f in files:
                ext = os.path.splitext(f)[1].lower()
                if ext in {".tif", ".tiff", ".png", ".jpg", ".jpeg"}:
                    file_path = os.path.join(root, f)

                    # Check corrupted
                    img = cv2.imread(file_path)
                    if img is None:
                        corrupted_files.append(file_path)
                        continue

                    # Image stats
                    h, w = img.shape[:2]
                    image_shapes.append((h, w))
                    brightness_values.append(img.mean())

                    # Check duplicates
                    with open(file_path, "rb") as f_img:
                        img_hash = hashlib.md5(f_img.read()).hexdigest()
                    if img_hash in hashes:
                        duplicates.append(file_path)
                    else:
                        hashes[img_hash] = file_path

                    sub_count += 1
                    class_total += 1

            subfolder_name = os.path.relpath(root, dataset_path)
            subfolder_counts[subfolder_name] = sub_count

        # Print table per class
        output += f"\nClass: {class_name}\n"
        table = [[sub, cnt] for sub, cnt in subfolder_counts.items()]
        output += tabulate(table, headers=["Subfolder", "Number of Images"], tablefmt="grid")
        output += f"\nTotal images in class '{class_name}': {class_total}\n\n"
        class_counts[class_name] = class_total
        total_images += class_total

    # Summary
    output += f"Total images in dataset: {total_images}\n"
    output += f"Corrupted images: {len(corrupted_files)}\n"
    if corrupted_files:
        for f in corrupted_files:
            output += f"⚠️ {f}\n"
    else:
        output += "No corrupted images found!\n"

    output += f"Duplicate images: {len(duplicates)}\n"
    if duplicates:
        for f in duplicates:
            output += f"⚠️ {f}\n"
    else:
        output += "No duplicate images found!\n"

    # Class distribution
    fig, ax = plt.subplots(figsize=(10,5))
    ax.bar(class_counts.keys(), class_counts.values())
    plt.xticks(rotation=45)
    ax.set_ylabel("Number of Images")
    ax.set_title(f"Class Distribution - {os.path.basename(dataset_path)}")
    figures.append(fig)

    # Image shapes
    if image_shapes:
        heights, widths = zip(*image_shapes)
        output += f"Image heights: min={min(heights)}, max={max(heights)}, mean={np.mean(heights):.2f}\n"
        output += f"Image widths: min={min(widths)}, max={max(widths)}, mean={np.mean(widths):.2f}\n"

        # Aspect ratio
        aspect_ratios = [w/h for h,w in image_shapes]
        fig, ax = plt.subplots()
        ax.hist(aspect_ratios, bins=20)
        ax.set_xlabel("Width / Height")
        ax.set_ylabel("Number of Images")
        ax.set_title(f"Aspect Ratio Distribution - {os.path.basename(dataset_path)}")
        figures.append(fig)

        # Brightness
        fig, ax = plt.subplots()
        ax.hist(brightness_values, bins=30)
        ax.set_xlabel("Mean Pixel Intensity")
        ax.set_ylabel("Number of Images")
        ax.set_title(f"Brightness Distribution - {os.path.basename(dataset_path)}")
        figures.append(fig)

    # Random samples
    for class_name in os.listdir(dataset_path):
        class_path = os.path.join(dataset_path, class_name)
        if not os.path.isdir(class_path):
            continue
        all_images = []
        for root, dirs, files in os.walk(class_path):
            for f in files:
                if os.path.splitext(f)[1].lower() in {".tif", ".tiff", ".png", ".jpg", ".jpeg"}:
                    all_images.append(os.path.join(root, f))
        
        if not all_images:
            continue
            
        sample_files = random.sample(all_images, min(3, len(all_images)))
        fig, axes = plt.subplots(1, len(sample_files), figsize=(10,3))
        if len(sample_files) == 1:
            axes = [axes]
        for i, fpath in enumerate(sample_files):
            img = cv2.imread(fpath)
            axes[i].imshow(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
            axes[i].set_title(os.path.basename(fpath))
            axes[i].axis('off')
        fig.suptitle(f"Random Samples - {class_name}")
        figures.append(fig)
        
    return output, figures

def compute_scanner_fingerprints():
    FLATFIELD_RESIDUALS_PATH = "d:/Digital Forensics Scanner/proceed_data/flatfield_residuals.pkl"
    FP_OUT_PATH = "d:/Digital Forensics Scanner/proceed_data/Flatfield/scanner_fingerprints.pkl"
    ORDER_NPY = "d:/Digital Forensics Scanner/proceed_data/Flatfield/fp_keys.npy"

    with open(FLATFIELD_RESIDUALS_PATH, "rb") as f:
        flatfield_residuals = pickle.load(f)

    scanner_fingerprints = {}
    st.write("🔄 Computing fingerprints from Flatfields...")
    for scanner, residuals in flatfield_residuals.items():
        if not residuals:
            continue
        stack = np.stack(residuals, axis=0)
        fingerprint = np.mean(stack, axis=0)
        scanner_fingerprints[scanner] = fingerprint

    with open(FP_OUT_PATH, "wb") as f:
        pickle.dump(scanner_fingerprints, f)

    fp_keys = sorted(scanner_fingerprints.keys())
    np.save(ORDER_NPY, np.array(fp_keys))
    st.write(f"✅ Saved {len(scanner_fingerprints)} fingerprints and fp_keys.npy")

def extract_prnu_features():
    FP_OUT_PATH = "d:/Digital Forensics Scanner/proceed_data/Flatfield/scanner_fingerprints.pkl"
    ORDER_NPY = "d:/Digital Forensics Scanner/proceed_data/Flatfield/fp_keys.npy"
    RES_PATH = "d:/Digital Forensics Scanner/proceed_data/official_wiki_residuals.pkl"
    FEATURES_OUT = "d:/Digital Forensics Scanner/proceed_data/features.pkl"

    with open(FP_OUT_PATH, "rb") as f:
        scanner_fingerprints = pickle.load(f)
    fp_keys = np.load(ORDER_NPY)

    def corr2d(a, b):
        a = a.astype(np.float32).ravel()
        b = b.astype(np.float32).ravel()
        a -= a.mean()
        b -= b.mean()
        denom = (np.linalg.norm(a) * np.linalg.norm(b))
        return float((a @ b) / denom) if denom != 0 else 0.0

    with open(RES_PATH, "rb") as f:
        residuals_dict = pickle.load(f)

    features, labels = [], []
    for dataset_name in ["Official", "Wikipedia"]:
        st.write(f"🔄 Computing PRNU features for {dataset_name} ...")
        for scanner, dpi_dict in tqdm(residuals_dict[dataset_name].items()):
            for dpi, res_list in dpi_dict.items():
                for res in res_list:
                    vec = [corr2d(res, scanner_fingerprints[k]) for k in fp_keys]
                    features.append(vec)
                    labels.append(scanner)

    with open(FEATURES_OUT, "wb") as f:
        pickle.dump({"features": features, "labels": labels}, f)
    st.write(f"✅ Saved features shape: {len(features)} x {len(features[0])}")

def extract_enhanced_features_func(): # Renamed to avoid conflict
    RES_PATH = "d:/Digital Forensics Scanner/proceed_data/official_wiki_residuals.pkl"
    ENHANCED_OUT = "d:/Digital Forensics Scanner/proceed_data/enhanced_features.pkl"

    def extract_enhanced_features(residual):
        fft_img = np.abs(fft2(residual))
        fft_img = fftshift(fft_img)
        h, w = fft_img.shape
        center_h, center_w = h//2, w//2
        low_freq = np.mean(fft_img[center_h-20:center_h+20, center_w-20:center_w+20])
        mid_freq = np.mean(fft_img[center_h-60:center_h+60, center_w-60:center_w+60]) - low_freq
        high_freq = np.mean(fft_img) - low_freq - mid_freq

        res_range = np.max(residual) - np.min(residual)
        if res_range > 0:
            residual_uint8 = (255 * (residual - np.min(residual)) / res_range).astype(np.uint8)
        else:
            residual_uint8 = np.zeros_like(residual, dtype=np.uint8)
        lbp = local_binary_pattern(residual_uint8, P=24, R=3, method='uniform')
        lbp_hist, _ = np.histogram(lbp, bins=26, range=(0, 25), density=True)

        grad_x = ndimage.sobel(residual, axis=1)
        grad_y = ndimage.sobel(residual, axis=0)
        grad_mag = np.sqrt(grad_x**2 + grad_y**2)
        texture_features = [
            np.std(residual),
            np.mean(np.abs(residual)),
            np.std(grad_mag),
            np.mean(grad_mag)
        ]

        return [low_freq, mid_freq, high_freq] + lbp_hist.tolist() + texture_features

    with open(RES_PATH, "rb") as f:
        residuals_dict = pickle.load(f)

    enhanced_features, enhanced_labels = [], []
    for dataset_name in ["Official", "Wikipedia"]:
        st.write(f"🔄 Extracting enhanced features for {dataset_name} ...")
        for scanner, dpi_dict in tqdm(residuals_dict[dataset_name].items()):
            for dpi, res_list in dpi_dict.items():
                for res in res_list:
                    feat = extract_enhanced_features(res)
                    enhanced_features.append(feat)
                    enhanced_labels.append(scanner)

    with open(ENHANCED_OUT, "wb") as f:
        pickle.dump({"features": enhanced_features, "labels": enhanced_labels}, f)
    st.write(f"✅ Enhanced features shape: {len(enhanced_features)} x {len(enhanced_features[0])}")
    st.write(f"✅ Saved enhanced features to {ENHANCED_OUT}")

def preprocess_and_compute_residuals():
    OFFICIAL_DIR = "d:/Digital Forensics Scanner/proceed_data/official"
    WIKI_DIR = "d:/Digital Forensics Scanner/proceed_data/Wikipedia"
    OUT_PATH = "d:/Digital Forensics Scanner/proceed_data/official_wiki_residuals.pkl"

    def to_gray(img):
        return cv2.cvtColor(img, cv2.COLOR_BGR2GRAY) if img.ndim == 3 else img

    def resize_to(img, size=(256, 256)):
        return cv2.resize(img, size, interpolation=cv2.INTER_AREA)

    def normalize_img(img):
        return img.astype(np.float32) / 255.0

    def denoise_wavelet(img):
        coeffs = pywt.dwt2(img, 'haar')
        cA, (cH, cV, cD) = coeffs
        cH[:] = 0
        cV[:] = 0
        cD[:] = 0
        return pywt.idwt2((cA, (cH, cV, cD)), 'haar')

    def compute_residual(img):
        denoised = denoise_wavelet(img)
        return img - denoised

    def process_single_image(fpath):
        img = cv2.imread(fpath, cv2.IMREAD_COLOR)
        if img is None:
            return None
        gray = to_gray(img)
        gray = resize_to(gray, (256, 256))
        gray = normalize_img(gray)
        return compute_residual(gray)

    def process_dataset(base_dir, dataset_name, residuals_dict):
        st.write(f"Recursively preprocessing {dataset_name} images...")
        dpi_dirs_to_process = []
        for root, dirs, files in os.walk(base_dir):
            if os.path.basename(root) in ['150', '300']:
                dpi_dirs_to_process.append(root)

        if not dpi_dirs_to_process:
            st.warning(f"Warning: No '150' or '300' DPI subfolders found in '{base_dir}'.")
            return

        for dpi_path in tqdm(dpi_dirs_to_process, desc=f"Processing {dataset_name} DPI folders"):
            dpi = os.path.basename(dpi_path)
            scanner_name = os.path.basename(os.path.dirname(dpi_path))

            files = [
                os.path.join(dpi_path, f) 
                for f in os.listdir(dpi_path) 
                if f.lower().endswith(('.tif', '.tiff', '.jpg', '.jpeg', '.png'))
            ]
            
            if not files:
                continue

            dpi_residuals = []
            with ThreadPoolExecutor(max_workers=8) as executor:
                futures = [executor.submit(process_single_image, f) for f in files]
                for fut in as_completed(futures):
                    res = fut.result()
                    if res is not None:
                        dpi_residuals.append(res)
            
            if scanner_name not in residuals_dict[dataset_name]:
                residuals_dict[dataset_name][scanner_name] = {}
            if dpi not in residuals_dict[dataset_name][scanner_name]:
                residuals_dict[dataset_name][scanner_name][dpi] = []
            
            residuals_dict[dataset_name][scanner_name][dpi].extend(dpi_residuals)

    residuals_dict = {"Official": {}, "Wikipedia": {}}
    process_dataset(OFFICIAL_DIR, "Official", residuals_dict)
    process_dataset(WIKI_DIR, "Wikipedia", residuals_dict)

    with open(OUT_PATH, "wb") as f:
        pickle.dump(residuals_dict, f)

    st.write(f"Saved Official + Wikipedia residuals (150 & 300 DPI separately) to {OUT_PATH}")

def predict_folder(folder_path, output_csv="hybrid_folder_results.csv"):
    IMG_SIZE = (256, 256)
    ART_DIR = "d:/Digital Forensics Scanner/proceed_data"
    FP_PATH = os.path.join(ART_DIR, "Flatfield/scanner_fingerprints.pkl")
    ORDER_NPY = os.path.join(ART_DIR, "Flatfield/fp_keys.npy")
    CKPT_PATH = os.path.join(ART_DIR, "scanner_hybrid_final.keras")

    hyb_model = tf.keras.models.load_model(CKPT_PATH, compile=False)

    with open(os.path.join(ART_DIR, "hybrid_label_encoder.pkl"), "rb") as f:
        le_inf = pickle.load(f)

    with open(os.path.join(ART_DIR, "hybrid_feat_scaler.pkl"), "rb") as f:
        scaler_inf = pickle.load(f)

    with open(FP_PATH, "rb") as f:
        scanner_fps_inf = pickle.load(f)

    fp_keys_inf = np.load(ORDER_NPY, allow_pickle=True).tolist()

    def corr2d(a, b):
        a = a.astype(np.float32).ravel(); b = b.astype(np.float32).ravel()
        a -= a.mean(); b -= b.mean()
        d = np.linalg.norm(a) * np.linalg.norm(b)
        return float((a @ b) / d) if d != 0 else 0.0

    def fft_radial_energy(img, K=6):
        f = np.fft.fftshift(np.fft.fft2(img))
        mag = np.abs(f)
        h, w = mag.shape; cy, cx = h//2, w//2
        yy, xx = np.ogrid[:h, :w]
        r = np.sqrt((yy - cy)**2 + (xx - cx)**2)
        rmax = r.max() + 1e-6
        bins = np.linspace(0, rmax, K+1)
        return [float(mag[(r >= bins[i]) & (r < bins[i+1])].mean() if ((r >= bins[i]) & (r < bins[i+1])).any() else 0.0) for i in range(K)]

    def lbp_hist_safe(img, P=8, R=1.0):
        rng = float(np.ptp(img))
        g = np.zeros_like(img, dtype=np.float32) if rng < 1e-12 else (img - img.min()) / (rng + 1e-8)
        g8 = (g * 255.0).astype(np.uint8)
        codes = local_binary_pattern(g8, P=P, R=R, method="uniform")
        hist, _ = np.histogram(codes, bins=np.arange(P+3), density=True)
        return hist.astype(np.float32).tolist()

    def preprocess_residual_pywt(path):
        img = cv2.imread(path, cv2.IMREAD_UNCHANGED)
        if img is None:
            raise ValueError(f"Cannot read {path}")
        if img.ndim == 3:
            img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        img = cv2.resize(img, IMG_SIZE, interpolation=cv2.INTER_AREA)
        img = img.astype(np.float32) / 255.0
        cA, (cH, cV, cD) = pywt.dwt2(img, 'haar')
        cH.fill(0); cV.fill(0); cD.fill(0)
        den = pywt.idwt2((cA, (cH, cV, cD)), 'haar')
        return (img - den).astype(np.float32)

    def make_feats_from_res(res):
        v_corr = [corr2d(res, scanner_fps_inf[k]) for k in fp_keys_inf]
        v_fft  = fft_radial_energy(res)
        v_lbp  = lbp_hist_safe(res)
        v = np.array(v_corr + v_fft + v_lbp, dtype=np.float32).reshape(1, -1)
        return scaler_inf.transform(v)

    def predict_scanner_hybrid(image_path):
        res = preprocess_residual_pywt(image_path)
        x_img = np.expand_dims(res, axis=(0,-1))
        x_feat = make_feats_from_res(res)
        prob = hyb_model.predict([x_img, x_feat], verbose=0)
        idx = int(np.argmax(prob))
        label = le_inf.classes_[idx]
        conf = float(prob[0, idx]*100)
        return label, conf

    exts=("*.tif","*.png","*.jpg","*.jpeg")
    image_files = []
    for ext in exts:
        image_files.extend(glob.glob(os.path.join(folder_path, "**", ext), recursive=True))
    st.write(f"Found {len(image_files)} images in {folder_path}")

    results = []
    for img_path in image_files:
        try:
            label, conf = predict_scanner_hybrid(img_path)
            results.append((img_path, label, conf))
            st.write(f"{img_path} -> {label} | {conf:.2f}%")
        except Exception as e:
            st.error(f"⚠️ Error {img_path}: {e}")

    with open(output_csv, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["Image", "Predicted_Label", "Confidence(%)"])
        writer.writerows(results)
    st.write(f"\n✅ Predictions saved to {output_csv}")
    return results

def main():
    '''Main function to run the Streamlit app.'''
    set_app_style()
    st.sidebar.title("AI TraceFinder")
    st.sidebar.markdown("Digital Scanner Forensics")

    pages = {
        "Home": home,
        "Prediction": prediction,
        "EDA": eda,
        "Feature Extraction": feature_extraction,
        "Testing": testing,
        "CNN Model": cnn_model
    }

    selection = st.sidebar.radio("Navigation", list(pages.keys()))

    # Render the selected page
    page = pages[selection]
    page()

if __name__ == "__main__":
    main()