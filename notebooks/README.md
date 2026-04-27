# Notebooks

These notebooks document the experimentation and model development process for this project.

They are **not** meant to be run end-to-end — they were used iteratively to test approaches,
compare results, and tune parameters. Final models were serialized and saved as `.pkl` files
in the `/models` directory.

## Contents

| Order | Notebook | Purpose | Outputs |
|-------|----------|---------|---------|
| 1 | `data_preprocessing.ipynb` | Combines raw CSVs (Telco, Subscription, Ecommerce), cleans and encodes features | `data/combined_cleaned_encoded.csv`, `data/combined_cleaned_unencoded.csv` |
| 2 | `feature_extraction.ipynb` | Extracts the 6-feature subset used for segmentation and encoders | `data/customer_features.csv` |
| 3 | `model_training.ipynb` | TDA feature engineering + ensemble model training (XGBoost, LightGBM, CatBoost) with SMOTE balancing | `backend/artifacts/super_ensemble.pkl`, `scaler.pkl`, `tda_node_centers.npy`, `tda_feature_columns.pkl`, `data/train_test_data/` |
| 4 | `save_label_mappings.ipynb` | Saves label encoders for categorical features (gender, payment method, industry) | `backend/artifacts/mappings.pkl` |
| 5 | `segmentation_and_clv.ipynb` | KMeans customer segmentation and CLV analysis | `backend/artifacts/segmentation_model.pkl`, `kmeans_scaler.pkl`, `kmeans_encoder.pkl`, `kmeans_feature_columns.pkl` |
| — | `explainability_analysis.ipynb` | SHAP and LIME analysis for model interpretability (reference only, not required to run the app) | — |

## Notes

- Outputs and plots may not reproduce exactly without the original dataset
- `.pkl` files in `/models` are the final artifacts used downstream