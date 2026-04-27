# ChurnGuard: Model Architecture

## Data

**3 industry datasets combined:**
| Dataset | Raw Rows | After Balancing |
|---|---|---|
| Telco | 7,043 | 7,043 (unchanged) |
| Subscription | 5,000 | 7,043 (oversampled + Gaussian noise) |
| Ecommerce | 50,000 | 7,043 (stratified undersample) |
| **Combined** | — | **21,129** |

**6 features used:** `gender`, `age`, `tenure`, `monthlycharges`, `paymentmethod`, `industry`

---

## Preprocessing

- **Encoding:** `LabelEncoder` on categorical features (gender, paymentmethod, industry)
- **Scaling:** `StandardScaler` on numeric features (age, tenure, monthlycharges)
- **Class imbalance:** `SMOTE` on training set → balanced to 10,054 per class (20,108 total)

---

## TDA (Topological Data Analysis)

Library: **KeplerMapper**

1. t-SNE reduces 6D → 2D lens
2. Mapper covers the space with `n_cubes=10, perc_overlap=0.3`, clusters inside each cube with `KMeans(n_clusters=4)`
3. Produces **340 nodes** → each sample gets a binary membership vector (1 if it belongs to that node)
4. These 340 binary features are appended to the 6 original → **346-dimensional final feature vector**

---

## Models Trained

| Model | Key Hyperparameters |
|---|---|
| XGBoost | `n_estimators=200, eval_metric=logloss` |
| LightGBM | `n_estimators=300, lr=0.05, num_leaves=31` |
| CatBoost | `iterations=300, lr=0.05, depth=6` |
| **Super Ensemble** | **Soft voting: XGB(w=2) + LGB(w=2) + CatBoost(w=3)** |

---

## Evaluation Metrics (Test set, n=4,226)

| Model | Accuracy | Precision | Recall | F1 |
|---|---|---|---|---|
| XGBoost | 0.6796 | 0.58 | 0.74 | 0.65 |
| LightGBM | 0.6817 | 0.60 | 0.67 | 0.63 |
| CatBoost | 0.6820 | 0.58 | 0.75 | 0.66 |
| **Super Ensemble** | **0.6822** | **0.59** | **0.72** | **0.65** |

**Confusion matrix (Super Ensemble):**
```
              Predicted No   Predicted Yes
Actual No       1643 (TN)       871 (FP)
Actual Yes       472 (FN)      1240 (TP)
```

---

## Segmentation

- **KMeans, k=3** on 18 features (3 numeric scaled + 15 one-hot encoded categoricals)
- Silhouette score: **0.30**
- CLV formula: `monthly charges × tenure × 1.2`

---

## Key Design Decisions

- **CatBoost weighted highest (3 vs 2)** in the ensemble — it had the best individual recall on churn class
- **TDA adds topological structure** — captures non-linear cluster boundaries invisible to standard features
- **SMOTE only on training data** — test set kept original distribution for honest evaluation
- **Cross-industry model** — one model generalises across Telecom, Subscription, and Ecommerce churn patterns
