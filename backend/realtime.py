import joblib
import numpy as np
import pandas as pd
import os
import shap


# Load artifacts
artifact_path = "backend/artifacts/"

model = joblib.load(artifact_path + "super_ensemble.pkl")
scaler = joblib.load(artifact_path + "scaler.pkl")
mappings = joblib.load(artifact_path + "mappings.pkl")            # dict of label encoders
tda_centers = np.load(artifact_path + "tda_node_centers.npy", allow_pickle=True)
feature_columns = joblib.load(os.path.join(artifact_path, "tda_feature_columns.pkl"))  # list of final columns for TDA (length 350)

kmeans_scaler = joblib.load(artifact_path + "kmeans_scaler.pkl")
kmeans_encoder = joblib.load(artifact_path + "kmeans_encoder.pkl")
kmeans_cols = joblib.load(artifact_path + "kmeans_feature_columns.pkl")
segmenter = joblib.load(artifact_path + "segmentation_model.pkl")

# From mappings (encoder values)
gender_map = {v: k for k, v in mappings["gender"].items()}
payment_map = {v: k for k, v in mappings["paymentmethod"].items()}
industry_map = {v: k for k, v in mappings["industry"].items()}



def encode_row(input_dict):
    encoded = []

    for col, mapping in mappings.items():
        value = input_dict[col]

        # unseen → -1
        encoded_value = mapping.get(value, -1)
        encoded.append(encoded_value)

    # append numeric
    for col in ["age", "tenure", "monthlycharges"]:
        encoded.append(float(input_dict[col]))

    return np.array(encoded, dtype=float)

def assign_tda_node(base_vector):
    distances = np.linalg.norm(tda_centers - base_vector, axis=1)
    return int(np.argmin(distances))

def preprocess_realtime(input_dict):
    
    # 1. MAIN MODEL ENCODING (350 features)
    encoded_and_numeric = encode_row(input_dict)

    n_cats = len(mappings)
    encoded_cats = encoded_and_numeric[:n_cats]
    numeric_vals = encoded_and_numeric[n_cats:]

    scaled_numeric = scaler.transform([numeric_vals])[0]

    base_for_tda = np.concatenate([encoded_cats, scaled_numeric])
    tda_node_index = assign_tda_node(base_for_tda)

    tda_onehot = np.zeros(len(tda_centers))
    tda_onehot[tda_node_index] = 1

    # final 350 features
    final_features = np.concatenate([encoded_cats, scaled_numeric, tda_onehot])

    # sanity
    if final_features.shape[0] != len(feature_columns):
        raise ValueError(
            f"Feature mismatch: got {final_features.shape[0]}, expected {len(feature_columns)}"
        )


    # FEATURES FOR KMEANS (18 features)
    kmeans_cat_cols = ['gender', 'paymentmethod', 'industry']

    kmeans_num_array = kmeans_scaler.transform(
        [[input_dict['age'], input_dict['tenure'], input_dict['monthlycharges']]]
    )[0]

    cat_array_for_kmeans = kmeans_encoder.transform(
        [[input_dict[c] for c in kmeans_cat_cols]]
    ).flatten()

    features_kmeans = np.concatenate([cat_array_for_kmeans, kmeans_num_array])

    
    # RAW EXPLAINER VECTOR (6 features)
    explainer_vector = np.array([
        input_dict["gender"],
        input_dict["paymentmethod"],
        input_dict["industry"],
        float(input_dict["age"]),
        float(input_dict["tenure"]),
        float(input_dict["monthlycharges"]),
    ])

    return final_features, features_kmeans, explainer_vector



def predict_realtime(input_dict):

    features_350, features_kmeans, explainer_vector = preprocess_realtime(input_dict)

    # model prediction
    probability = float(model.predict_proba([features_350])[0][1])
    prediction = int(model.predict([features_350])[0])

    # segmentation
    segment = int(segmenter.predict([features_kmeans])[0])
    
    # CLV calculation
    monthly = float(input_dict.get("monthlycharges", 0))
    tenure = float(input_dict.get("tenure", 0))
    clv_value = monthly * tenure * 1.2
    
    # CLV thresholds
    if clv_value < 1000:
        clv_label = "Low"
    elif clv_value < 3000:
        clv_label = "Medium"
    else:
        clv_label = "High"
    
    # Segment meaning
    # 0 = Low value group
    # 1 = Medium value group
    # 2 = High value group
    
    # Priority:
    #  CLV as primary signal 
    # segment to adjust classification when CLV is borderline
    
    if clv_label == "High":
        final_value = "High Value"
    
    elif clv_label == "Medium":
        if segment == 2:          # high-value segment
            final_value = "High Value"
        elif segment == 1:        # medium-value segment
            final_value = "Medium Value"
        else:                     # seg 0 + medium CLV
            final_value = "Low Value"
    
    else:  # clv_label == "Low"
        if segment == 2:          # high-value segment but low CLV
            final_value = "Medium Value"
        elif segment == 1:        # medium segment
            final_value = "Low Value"
        else:                     # seg 0
            final_value = "Low Value"


    
    # Recommended action
    if probability >= 0.7:
        if clv_value > 2000:
            action = "50% off for next 3 months" 
        else:
            action = "Offer 10% Discount" 
    
    elif probability >= 0.5:
        if clv_value > 2000:
            action = "Offer Loyalty Bonus (Free Month)"
        else:
            action = "Automated 'We miss you' email"
    
    else: # Low churn probability
        if clv_value > 2000:
            action = "Request Referral / VIP Event Invite"
        else:
            action = "Suggest Premium Plan"
    
    # SHAP
    shap_top_features = compute_shap_realtime(model, features_350, input_dict, final_value)
    
    return {
        "prediction": prediction,
        "probability": probability,
        "segment": final_value,
        "clv": clv_value,
        "recommended_action": action,
        "shap_explanation": shap_top_features
    }



# SHAP EXPLAINABILITY
def compute_shap_realtime(model, final_feature_vector, input_dict, final_value):
    
    # background: mean of 350 features
    background = np.zeros((1, len(final_feature_vector))) 
    
    def predict_fn(X):
        return model.predict_proba(X)[:, 1]

    explainer = shap.KernelExplainer(predict_fn, background)

    # convert input to 2D numpy array
    X_input = np.reshape(final_feature_vector, (1, -1))
    shap_values = explainer.shap_values(X_input)[0]

    abs_vals = np.abs(shap_values)
    
    sorted_idx = abs_vals.argsort()[::-1]

    top_features = []
    
    for idx in sorted_idx:
        feature_name = feature_columns[idx] # 350 feature column names
        
        # Skip tda
        if "tda" in str(feature_name): 
            continue

        top_features.append({
            "feature": feature_name,
            "shap_value": float(shap_values[idx])
        })

        if len(top_features) >= 10:
            break


    # NATURAL LANGUAGE EXPLANATION    
    column_names = [
        "gender",           # 0
        "paymentmethod",    # 1
        "industry",          # 2
        "age",              # 3
        "tenure",           # 4
        "monthlycharges"  # 5
    ]

    # PREDICT PROBABILITY
    prob = float(model.predict_proba([final_feature_vector])[0][1])
    
    explanations = []
    industry_name = ""
    name = ""

    # NATURAL LANGUAGE 
    for feat in top_features:
        
        if feat["feature"] >= 0:
            col = int(feat["feature"])
            if col > 5:
                continue
            name = column_names[col]
        else:
            continue

        
        raw_value = input_dict.get(name)

        if name == "gender":
            continue            

        elif name == "age":
            if raw_value < 35:
                explanations.append(f"is a young customer")
            elif raw_value < 55:
                explanations.append(f"is a middle-aged customer")
            else:
                explanations.append(f"is a senior customer")           

        elif name == "tenure":
            if raw_value < 15:
                explanations.append(f"is a short-tenure customer")
            elif raw_value < 35:
                explanations.append(f"is a medium-tenure customer")
            else:
                explanations.append(f"is a long-tenure customer")            

        elif name == "monthlycharges":
            if raw_value < 40:
                explanations.append(f"has low monthly charges")
            elif raw_value < 90:
                explanations.append(f"has moderate monthly charges")
            else:
                explanations.append(f"has high monthly charges")
            
        elif name == "paymentmethod":
            decoded = raw_value 
            explanations.append(f"uses {decoded}")

        elif name == "industry":
            industry_name = raw_value

        else:
            explanations.append(name.replace("_"," "))

    # Combine reasons
    if len(explanations) > 1:
        reasons = ", ".join(explanations[:-1]) + ", and " + explanations[-1]

    # FINAL SENTENCE 
    sentence = (
        f"Customer from the {industry_name} industry is predicted to churn "
        f"with a probability of {prob:.2f}. "
        f"This is mainly because the customer {reasons}. Therefore, this is a {final_value.lower()} customer."
    )
    
    return sentence
    

