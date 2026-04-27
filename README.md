# ChurnGuard: Predicting and Preventing Customer Loss

A cross-industry customer churn prediction system built with a TDA-enhanced gradient boosting ensemble and SHAP-based explanations, served through a real-time Streamlit dashboard.

---

## Quick Start

```bash
pip install -r requirements.txt
streamlit run streamlit_app.py
```

Two pages are available from the sidebar:

| Page | Description |
|------|-------------|
| **Real-Time Dashboard** | Auto-refreshes every 10 seconds with a randomly generated customer and live churn prediction |
| **Manual Prediction** | Enter customer details manually and get an instant churn forecast with explanation |

---

## How It Works

Each prediction runs through five stages:

1. **Encoding** — Categorical and numerical inputs are encoded and scaled
2. **TDA feature augmentation** — A topological representation of the input is appended to the base features
3. **Ensemble prediction** — A soft-voting ensemble of XGBoost, LightGBM, and CatBoost outputs a churn probability
4. **Segmentation & CLV** — A KMeans model assigns the customer to a value segment (Low / Medium / High)
5. **Explanation** — SHAP values are converted into a plain-English reason for the prediction

---

## Results

Evaluated on a held-out test set (n = 4,226):

| Model | Accuracy | Precision | Recall | F1 |
|---|---|---|---|---|
| XGBoost | 0.68 | 0.58 | 0.74 | 0.65 |
| LightGBM | 0.68 | 0.60 | 0.67 | 0.63 |
| CatBoost | 0.68 | 0.58 | 0.75 | 0.66 |
| **Super Ensemble** | **0.68** | **0.59** | **0.72** | **0.65** |

---

## Data

Three public datasets combined into a unified cross-industry training set:

- [Telco Customer Churn](https://www.kaggle.com/datasets/blastchar/telco-customer-churn)
- [E-Commerce Churn](https://www.kaggle.com/datasets/surajbhandari527/ecommerce-churn-data-for-churn-prediction-models/data)
- [Subscription](https://www.kaggle.com/datasets/akashanandt/streaming-service-data)(Streaming Service Data on Kaggle)

---

## Tech Stack

| Component | Library |
|-----------|---------|
| App framework | Streamlit |
| ML models | XGBoost, LightGBM, CatBoost |
| Ensemble & preprocessing | scikit-learn |
| Class imbalance | imbalanced-learn (SMOTE) |
| Topological Data Analysis | KeplerMapper |
| Explainability | SHAP, LIME |
| Data handling | pandas, numpy |

---

## Team

Built by **Team ChurnGuard**:

- Sohail Mohammed
- Jessica Thomas
- Krish Khatri
- Karishma Doshi
