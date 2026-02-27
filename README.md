# Predicting In-Hospital Mortality using MIMIC-III Data 
### *Multimodal 24h ICU Model*

This project builds an early ICU risk model to predict in-hospital mortality for ICU patients using the MIMIC-III dataset. We use the first 24 hours after ICU admission ([intime, intime + 24h)) and combine:
	•	Structured features: 24h vitals + labs (aggregated statistics)
	•	Unstructured text: 24h nursing notes encoded with Bio_ClinicalBERT (768-D embeddings)

The evaluation uses patient-level group splitting (SUBJECT_ID) to avoid leakage across multiple ICU stays per patient, and reports both discrimination and calibration metrics (AUROC, AUPRC, Brier, ECE).

- **Authors:** Jingyi(Jenny) Chen
- **Full Report:** [/report/predict_mortality.pdf](./report/predict_mortality.pdf)

  
- ** Key Highlights **

* **Multimodal Fusion:** Combines time-series vitals/labs with NLP-based nursing note embeddings.
* **Clinical-Specific NLP:** Leverages `Bio_ClinicalBERT` for specialized medical text representation (768-D).
* **Robust Evaluation:** Implements **Patient-level Group Splitting** (`SUBJECT_ID`) to prevent data leakage—ensuring the model generalizes to new patients.
* **Interpretability:** Uses **SHAP** values to decode model decisions for clinical transparency.
  

---

## 📊 Methodology & Tech Stack

### 1. Data Pipeline
* **Cohort Construction:** Built a master cohort anchored at `ICUSTAY_ID` by merging ICUSTAYS, ADMISSIONS, and PATIENTS tables.
* **Structured Engineering:** Aggregating 24h vitals (`CHARTEVENTS`) and labs (`LABEVENTS`) using statistics (mean, min, max, std, count).
* **Unstructured NLP:** Nursing notes aligned to the first 24h, processed via a sliding window + mean-pooling through **Bio_ClinicalBERT**.

### 2. Tech Stack
* **Modeling:** `XGBoost`, `Scikit-learn`
* **Deep Learning/NLP:** `PyTorch`, `Hugging Face Transformers`
* **Analytics:** `Pandas`, `NumPy`, `Matplotlib`, `SHAP`

---

## 📈 Key Results

The multimodal approach consistently outperforms the structured-only model, especially in calibration and cases where clinical notes are rich.

### 1. Overall Performance (Full Cohort)
| Setting | AUROC | AUPRC | Brier Score | ECE (Calibration) |
| :--- | :---: | :---: | :---: | :---: |
| **Structured Only** | 0.8984 | 0.6126 | 0.0631 | 0.0114 |
| **Multimodal (Fusion)** | **0.9017** | **0.6228** | **0.0622** | **0.0087** |

> **Insight:** Adding ClinicalBERT embeddings provides a significant boost in **Calibration (ΔECE -23.6%)**, making the predicted probabilities far more reliable for real-world clinical decision-making.

### 2. Text-Available Subset (59.6% of stays)
When early nursing notes are present, the multimodal gains are substantially amplified.
| Setting | AUROC | AUPRC | Brier Score | ECE |
| :--- | :---: | :---: | :---: | :---: |
| Structured Only | 0.9106 | 0.6238 | 0.0597 | 0.0101 |
| **Multimodal (Fusion)** | **0.9206** | **0.6602** | **0.0563** | 0.0113 |

> **Key Insights:** > * **Multimodal Lift:** In the text-available subset, adding ClinicalBERT embeddings yields a significant boost (**+3.64% AUPRC** and **+1.00% AUROC**).
> * **Clinical Reliability:** The multimodal model consistently improves calibration (ECE) across the full cohort, ensuring that predicted mortality probabilities align closely with actual clinical outcomes.

## Repository Structure

```
.
├── 📓 code/
│   ├── 01_clean_build_cohort.ipynb  # data merging & cleaning
│   ├── 02_clean_vitals_24h.ipynb    # Vitals feature engineering
│   ├── 03_clean_labs_24h.ipynb      # Labs feature engineering
│   ├── 04_EDA.ipynb                 # Exploratory Data Analysis
│   ├── 05_notes_24h_nursing.ipynb   # Text alignment & preprocessing
│   ├── 06_clinbert_embeddings.ipynb # Bio_ClinicalBERT embedding logic
│   ├── 07_structured_modeling.ipynb # Baseline XGBoost models
│   └── 08_multimodal_modeling.ipynb # Late fusion experiments
├── figures/
│   ├── roc_overall.png
│   ├── pr_overall.png
│   ├── calib_overall.png
│   ├── pr_has_text1.png
│   ├── calib_has_text1.png
│   ├── shap_bar.png
│   └── shap_beeswarm.png
├── report/
│   └── predict_mortality.pdf
├── requirements.txt
└── README.md     
```
## How to Replicate

1.  **Code:** The full analysis is available in the Jupyter Notebooks in the `/code/` folder.
2.  **Data:** The project uses the [MIMIC-III dataset](https://physionet.org/content/mimiciii/1.4/), which requires access permission from PhysioNet. The code assumes the raw data tables are available.
