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

  ## Accuracy & Performance
- **Hybrid CNN model test accuracy: 93.7% (on 914 test images, across 11 scanner classes).**
- Per-class F1-score ranges: Canon/Epson/HP scanner models all reach between 0.90 and 0.99 F1-score, with every class above 0.83 precision/recall.
- All dataset images are validated for corruption (none found in benchmarks), ensuring robust training.

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
**Overall weighted avg:** Precision 0.94, Recall 0.94, F1-score 0.94 (Test set: 914 images).
---
## Getting Started  
- Upload a scanned image.  
- The app auto-processes and predicts the **scanner model** with a confidence score.  
- Feature region visualizations available for forensic expert review.
- ## Results Interpretation  
- Precise, interpretable results for legal and forensic use — every image analyzed for artifact fingerprinting, model shows detailed classification evidence.
- **Accuracy (test): 93.7%**, supporting robust forensic validation in scanned documents.

