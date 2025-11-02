#  AI TraceFinder — Forensic Scanner Identification  

##  Overview  
AI TraceFinder is a forensic machine learning platform that identifies the **source scanner device** used to digitize a document or image. Each scanner (brand/model) introduces unique **noise, texture, and compression artifacts** that serve as a fingerprint. By analyzing these patterns, AI TraceFinder enables **fraud detection, authentication, and forensic validation** in scanned documents.  

---

##  Goals & Objectives  
- Collect and label scanned document datasets from multiple scanners  
- Robust preprocessing (resize, grayscale, normalize, denoise)  
- Extract scanner-specific features (noise, FFT, PRNU, texture descriptors)  
- Train classification models (ML + CNN)  
- Apply explainability tools (Grad-CAM, SHAP)  
- **Deploy an interactive app for scanner source identification**  
- Deliver **accurate, interpretable results** for forensic and legal use cases  

---

##  Methodology 
1. **Data Collection & Labeling**  
   - Gather scans from 3–5 scanner models/brands  
   - Create a structured, labeled dataset  

2. **Preprocessing**  
   - Resize, grayscale, normalize  
   - Optional: denoise to highlight artifacts  

3. **Feature Extraction**  
   - PRNU patterns, FFT, texture descriptors (LBP, edge features)  

4. **Model Training**  
   - Baseline ML: SVM, Random Forest, Logistic Regression  
   - Deep Learning: CNN with augmentation  

5. **Evaluation & Explainability**  
   - Metrics: Accuracy, F1-score, Confusion Matrix  
   - Interpretability: Grad-CAM, SHAP feature maps  

6. **Deployment**  
   - Streamlit app → upload scanned image → predict scanner model  
   - Display confidence score and key feature regions  

---

##  Actionable Insights for Forensics  
- **Source Attribution:** Identify which scanner created a scanned copy of a document.  
- **Fraud Detection:** Detect forgeries where unauthorized scanners were used.  
- **Legal Verification:** Validate whether scanned evidence originated from approved devices.  
- **Tamper Resistance:** Differentiate between authentic vs. tampered scans.  
- **Explainability:** Provide visual evidence of how classification was made.  

---

##  Architecture (Conceptual)  
Input ➜ Preprocessing ➜ Feature Extraction + Modeling ➜ Evaluation & Explainability ➜ Prediction App  

---

 ## 🎯 Accuracy & Performance  

The **Hybrid CNN model** was trained using a mix of **image-based inputs** and **handcrafted statistical features**.  
The dataset consisted of:  
- **Training set:** 3,654 document scans  
- **Test set:** 914 document scans  
Each image was resized to **256×256 (grayscale)** and paired with **27 handcrafted features** across **11 scanner classes**.

**Hybrid train : (3654, 256, 256, 1) (3654, 27) (3654, 11)**

**Hybrid test : (914, 256, 256, 1) (914, 27) (914, 11)**


After 50 epochs of training, the hybrid model achieved an impressive **test accuracy of 93.65%**, proving effective across diverse scanner brands and image formats.  
All images were verified for data integrity before model training — no corrupted files were detected.

| Scanner Model   | Precision | Recall | F1-score | Support |
|-----------------|-----------|--------|----------|---------|
| Canon120-1      | 0.91      | 0.89   | 0.90     | 83      |
| Canon120-2      | 0.84      | 0.83   | 0.84     | 83      |
| Canon220        | 0.89      | 0.92   | 0.90     | 83      |
| Canon9000-1     | 0.95      | 0.87   | 0.91     | 83      |
| Canon9000-2     | 0.88      | 0.95   | 0.91     | 82      |
| EpsonV370-1     | 0.99      | 0.95   | 0.97     | 83      |
| EpsonV370-2     | 0.95      | 0.99   | 0.97     | 84      |
| EpsonV39-1      | 0.94      | 0.96   | 0.95     | 83      |
| EpsonV39-2      | 0.96      | 0.95   | 0.96     | 84      |
| EpsonV550       | 1.00      | 0.99   | 0.99     | 83      |
| HP              | 0.99      | 1.00   | 0.99     | 83      |

**Overall Weighted Performance:**  
- **Precision:** 0.94  
- **Recall:** 0.94  
- **F1-score:** 0.94  
- **Test Accuracy:** 93.65%  

🧠 *Insight:*  
The hybrid model effectively combines CNN-learned visual patterns with statistical handcrafted features, enabling precise scanner identification even among closely related device models.
**<img width="964" height="745" alt="image" src="https://github.com/user-attachments/assets/fbca0f9f-3f0f-4f5e-9c68-f0ce2e48a166" />**

---
**Prerequisites**
- Python 3.10  
- pip install Requirement.txt file
- (Optional) GPU & CUDA if training on GPU
- # Create and activate virtual environment (recommended)
python -m venv .venv
# Windows
.venv\Scripts\activate
# Install dependencies
pip install -r Requirements.txt
We used a public scanner dataset (link / openmfc). You can either download manually or use the helper script below.
 # https://www.nist.gov/
 # Run the demo (Streamlit)
 **streamlit run streamlit_app.py**
 # then open http://localhost:8501
 # Evaluate using pre-trained model
 # Train the model from scratch
Maintainer: Amit Kumar  
Email: amitkushwaha200215@gmail.com
GitHub: https://github.com/amitkushwaha001/



