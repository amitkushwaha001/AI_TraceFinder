import os
import cv2
import numpy as np
import pickle
from tqdm import tqdm
import pywt
from concurrent.futures import ThreadPoolExecutor, as_completed

# Paths (adjust according to your folder structure)
OFFICIAL_DIR = "D:\Digital Forensics Scanner/proceed_data/official"
WIKI_DIR = "D:\Digital Forensics Scanner/proceed_data/Wikipedia"
OUT_PATH = "D:\Digital Forensics Scanner/proceed_data/official_wiki_residuals.pkl"

# =====================
# Preprocessing helpers
# =====================
def to_gray(img):
    """Convert image to grayscale if it's color."""
    return cv2.cvtColor(img, cv2.COLOR_BGR2GRAY) if img.ndim == 3 else img

def resize_to(img, size=(256, 256)):
    """Resize image to a fixed size."""
    return cv2.resize(img, size, interpolation=cv2.INTER_AREA)

def normalize_img(img):
    """Normalize image to range [0,1]."""
    return img.astype(np.float32) / 255.0

def denoise_wavelet(img):
    """Apply wavelet denoising (Haar) and remove high-frequency details."""
    coeffs = pywt.dwt2(img, 'haar')
    cA, (cH, cV, cD) = coeffs
    cH[:] = 0
    cV[:] = 0
    cD[:] = 0
    return pywt.idwt2((cA, (cH, cV, cD)), 'haar')

def compute_residual(img):
    """Compute residual by subtracting denoised image from original."""
    denoised = denoise_wavelet(img)
    return img - denoised

def process_single_image(fpath):
    """Preprocess a single image and return its residual."""
    img = cv2.imread(fpath, cv2.IMREAD_COLOR)
    if img is None:
        return None
    gray = to_gray(img)
    gray = resize_to(gray, (256, 256))
    gray = normalize_img(gray)
    return compute_residual(gray)

# =====================
# Dataset processing
# =====================
def process_dataset(base_dir, dataset_name, residuals_dict):
    print(f"Recursively preprocessing {dataset_name} images...")

    # 1. Find all '150' and '300' DPI subfolders
    dpi_dirs_to_process = []
    for root, dirs, files in os.walk(base_dir):
        if os.path.basename(root) in ['150', '300']:
            dpi_dirs_to_process.append(root)

    if not dpi_dirs_to_process:
        print(f"Warning: No '150' or '300' DPI subfolders found in '{base_dir}'.")
        return

    # 2. Process found folders with a progress bar
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
        # Parallel processing
        with ThreadPoolExecutor(max_workers=8) as executor:
            futures = [executor.submit(process_single_image, f) for f in files]
            for fut in as_completed(futures):
                res = fut.result()
                if res is not None:
                    dpi_residuals.append(res)
        
        # Add processed data to dictionary
        if scanner_name not in residuals_dict[dataset_name]:
            residuals_dict[dataset_name][scanner_name] = {}
        if dpi not in residuals_dict[dataset_name][scanner_name]:
            residuals_dict[dataset_name][scanner_name][dpi] = []
        
        residuals_dict[dataset_name][scanner_name][dpi].extend(dpi_residuals)

# =====================
# Main Execution
# =====================
residuals_dict = {"Official": {}, "Wikipedia": {}}

# Process official and Wikipedia datasets
process_dataset(OFFICIAL_DIR, "Official", residuals_dict)
process_dataset(WIKI_DIR, "Wikipedia", residuals_dict)

# Save residuals to pickle file
with open(OUT_PATH, "wb") as f:
    pickle.dump(residuals_dict, f)

print(f"Saved Official + Wikipedia residuals (150 & 300 DPI separately) to {OUT_PATH}")
