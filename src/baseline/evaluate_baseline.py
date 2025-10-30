import matplotlib
matplotlib.use('Agg')
import numpy as np
import pandas as pd
import joblib
import pickle
from sklearn.metrics import classification_report, confusion_matrix
import seaborn as sns
import matplotlib.pyplot as plt
import os
from skimage.feature import local_binary_pattern
from scipy.fft import fft2, fftshift

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

def make_feats_from_res(res, scanner_fps, fp_keys):
    v_corr = [corr2d(res, scanner_fps[k]) for k in fp_keys]
    v_fft  = fft_radial_energy(res)
    v_lbp  = lbp_hist_safe(res)
    return v_corr + v_fft + v_lbp

def evaluate_model(model_path, name, save_dir="d:/Digital Forensics Scanner/results"):

    # Load residuals and fingerprints (cache if possible)
    print("Loading model artifacts...")
    with open("d:/Digital Forensics Scanner/proceed_data/official_wiki_residuals.pkl", "rb") as f:
        residuals_dict = pickle.load(f)
    with open("d:/Digital Forensics Scanner/proceed_data/Flatfield/scanner_fingerprints.pkl", "rb") as f:
        scanner_fingerprints = pickle.load(f)
    fp_keys = np.load("d:/Digital Forensics Scanner/proceed_data/Flatfield/fp_keys.npy", allow_pickle=True).tolist()

    # Sample up to 200 images per class for speed
    import random
    from concurrent.futures import ThreadPoolExecutor, as_completed
    features = []
    labels = []
    print("Extracting features for evaluation (ultra-fast mode)...")
    for dataset_name in ["Official", "Wikipedia"]:
        for scanner, dpi_dict in residuals_dict[dataset_name].items():
            for dpi, res_list in dpi_dict.items():
                sample_size = min(50, len(res_list))
                sampled_res = random.sample(res_list, sample_size) if len(res_list) > sample_size else res_list
                # Parallel feature extraction
                with ThreadPoolExecutor(max_workers=8) as executor:
                    future_to_res = {executor.submit(make_feats_from_res, res, scanner_fingerprints, fp_keys): res for res in sampled_res}
                    for future in as_completed(future_to_res):
                        features.append(future.result())
                        labels.append(scanner)

    X = np.array(features)
    y = np.array(labels)

    # Load the same label encoder used during training
    label_encoder = joblib.load("d:/Digital Forensics Scanner/models/baseline_label_encoder.pkl")
    scaler = joblib.load("d:/Digital Forensics Scanner/models/scaler.pkl")
    model = joblib.load(model_path)

    # Transform labels to numerical values using the same encoder
    y_encoded = label_encoder.transform(y)

    X_scaled = scaler.transform(X)
    y_pred_encoded = model.predict(X_scaled)
    
    # Convert predictions back to labels for report
    y_pred = label_encoder.inverse_transform(y_pred_encoded)

    print(f"\n=== {name} Evaluation ===")
    # Generate and save classification report
    report_dict = classification_report(
        y, y_pred,
        labels=label_encoder.classes_,
        target_names=label_encoder.classes_,
        output_dict=True
    )
    df_report = pd.DataFrame(report_dict).transpose()
    report_path = os.path.join(save_dir, f"{name.replace(' ', '_')}_classification_report.csv")
    df_report.to_csv(report_path)
    print(f"Classification report saved to: {report_path}")

    # Generate and save confusion matrix
    cm = confusion_matrix(y, y_pred, labels=label_encoder.classes_)
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt="d",
                xticklabels=label_encoder.classes_,
                yticklabels=label_encoder.classes_,
                cmap="Blues")
    plt.title(f"{name} Confusion Matrix")
    plt.xlabel("Predicted")
    plt.ylabel("True")

    os.makedirs(save_dir, exist_ok=True)
    save_path = os.path.join(save_dir, f"{name.replace(' ', '_')}_confusion_matrix.png")
    plt.savefig(save_path, dpi=300, bbox_inches="tight")
    print(f"Confusion matrix saved to: {save_path}")

    plt.close()

if __name__ == "__main__":
    evaluate_model("d:/Digital Forensics Scanner/models/random_forest.pkl", "Random Forest")
    evaluate_model("d:/Digital Forensics Scanner/models/svm.pkl", "SVM")
