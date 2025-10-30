'''
Trains baseline models (Random Forest and SVM) on enhanced features.
'''
import pandas as pd
import numpy as np
import joblib
import pickle
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from tqdm import tqdm
from skimage.feature import local_binary_pattern
from scipy.fft import fft2, fftshift

# --- Feature Extraction --- #

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

def train_models():
    '''Trains and saves the baseline models.'''
    print("Loading data...")
    
    # Load residuals
    with open("d:/Digital Forensics Scanner/proceed_data/official_wiki_residuals.pkl", "rb") as f:
        residuals_dict = pickle.load(f)

    # Load scanner fingerprints
    with open("d:/Digital Forensics Scanner/proceed_data/Flatfield/scanner_fingerprints.pkl", "rb") as f:
        scanner_fingerprints = pickle.load(f)
    
    fp_keys = np.load("d:/Digital Forensics Scanner/proceed_data/Flatfield/fp_keys.npy", allow_pickle=True).tolist()

    features = []
    labels = []

    print("Extracting features...")
    for dataset_name in ["Official", "Wikipedia"]:
        for scanner, dpi_dict in tqdm(residuals_dict[dataset_name].items(), desc=f"Processing {dataset_name}"):
            for dpi, res_list in dpi_dict.items():
                for res in res_list:
                    feats = make_feats_from_res(res, scanner_fingerprints, fp_keys)
                    features.append(feats)
                    labels.append(scanner)

    X = np.array(features)
    y = np.array(labels)

    print(f"Feature matrix shape: {X.shape}")

    # Encode labels
    le = LabelEncoder()
    y_encoded = le.fit_transform(y)

    # Split data
    X_train, X_test, y_train, y_test = train_test_split(
        X, y_encoded, test_size=0.2, stratify=y_encoded, random_state=42
    )

    # Scale features
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)

    print("Training Random Forest model...")
    rf = RandomForestClassifier(n_estimators=300, random_state=42, n_jobs=-1)
    rf.fit(X_train_scaled, y_train)

    print("Training SVM model...")
    svm = SVC(kernel="rbf", C=10, gamma="scale", probability=True)
    svm.fit(X_train_scaled, y_train)

    # Save models and artifacts
    joblib.dump(rf, "models/random_forest.pkl")
    joblib.dump(svm, "models/svm.pkl")
    joblib.dump(scaler, "models/scaler.pkl")
    joblib.dump(le, "models/baseline_label_encoder.pkl") # Save the label encoder

    print("Models, scaler, and label encoder trained and saved successfully!")

if __name__ == "__main__":
    train_models()
