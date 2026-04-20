"""
train_and_save.py
─────────────────
Run once locally to train all models and save them.
Command: python train_and_save.py

Saves everything to models/
"""

import pandas as pd
import numpy as np
import joblib
import os
import warnings
warnings.filterwarnings("ignore")

os.makedirs("models", exist_ok=True)

# ══════════════════════════════════════════════════════════════
# LOAD DATA
# ══════════════════════════════════════════════════════════════
cleaned    = pd.read_csv("Dataset/Augmented_Cleaned_Data.csv")
cat_data   = pd.read_csv("Dataset/Augmented_Categorical_Data.csv")
scored_cln = pd.read_csv("Dataset/Scored_Cleaned_data.csv")

for df in [cleaned, cat_data, scored_cln]:
    df.drop(columns=[c for c in df.columns if "Unnamed" in c], inplace=True)

print(f"✅ Data loaded")

# ══════════════════════════════════════════════════════════════
# 1. NEURAL NETWORK — Monthly Spend Tier (1-10)
# ══════════════════════════════════════════════════════════════
from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout
from tensorflow.keras.utils import to_categorical

NN_DROP  = ["Monthly_Spend", "Group"]
X_nn     = cleaned.drop(columns=NN_DROP).copy().fillna(0).astype(float)
y_nn     = cleaned["Monthly_Spend"].values - 1

joblib.dump(X_nn.columns.tolist(), "models/nn_feature_cols.joblib")

mms = MinMaxScaler()
mms.fit(cleaned[["Unplanned_Purchases","Peer_Influence","Finance_Confidence"]])
joblib.dump(mms, "models/mms_scaler.joblib")

y_cat    = to_categorical(y_nn, num_classes=10)
nn_model = Sequential([
    Dense(60, activation='relu', input_shape=(X_nn.shape[1],)),
    Dropout(0.45),
    Dense(30, activation='relu'),
    Dense(20, activation='relu'),
    Dense(10, activation='softmax'),
])
nn_model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
nn_model.fit(X_nn.values, y_cat, epochs=40, batch_size=50, validation_split=0.15, verbose=0)
nn_model.save("models/nn_model.h5")
print("✅ Neural Network saved")

# ══════════════════════════════════════════════════════════════
# 2. RANDOM FOREST — Risk Score (1-100)
# ══════════════════════════════════════════════════════════════
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import LabelEncoder, StandardScaler
from scipy.spatial.distance import cdist

budget_cols = ["Budget_FoodDining","Budget_Travel","Budget_Fashion",
               "Budget_Subscriptions","Budget_Entertainment"]
cat_data[budget_cols] = cat_data[budget_cols].replace("Answer","Yes").fillna("No")
cat_data["Expenditure_Graph"] = cat_data["Expenditure_Graph"].astype(str)

X_rf = cat_data.drop(columns=["Monthly_Spend"], errors='ignore').copy()
y_rf = cat_data["Group"].values

le_encoders = {}
for col in X_rf.columns:
    if not pd.api.types.is_numeric_dtype(X_rf[col]):
        le = LabelEncoder()
        X_rf[col] = le.fit_transform(X_rf[col].astype(str).fillna("Missing"))
        le_encoders[col] = le

X_rf = X_rf.fillna(X_rf.median())
rf_scaler   = StandardScaler()
X_rf_scaled = rf_scaler.fit_transform(X_rf)

safe_ref    = X_rf_scaled[y_rf == 1].mean(axis=0)
rf          = RandomForestClassifier(n_estimators=150, random_state=42)
rf.fit(X_rf_scaled, y_rf)
importances = rf.feature_importances_

weighted_X    = X_rf_scaled * importances
weighted_safe = safe_ref * importances
rf_distances  = cdist(weighted_X, [weighted_safe], metric='euclidean').flatten()

score_scaler = MinMaxScaler(feature_range=(1, 100))
score_scaler.fit(rf_distances.reshape(-1,1))

joblib.dump(le_encoders,           "models/le_encoders.joblib")
joblib.dump(rf_scaler,             "models/rf_scaler.joblib")
joblib.dump(importances,           "models/rf_importances.joblib")
joblib.dump(safe_ref,              "models/rf_safe_reference.joblib")
joblib.dump(score_scaler,          "models/rf_score_scaler.joblib")
joblib.dump(X_rf.columns.tolist(), "models/rf_feature_cols.joblib")
print("✅ RF Risk Scorer saved")

# ══════════════════════════════════════════════════════════════
# 3. KMEANS k=20 + FUSION
# ══════════════════════════════════════════════════════════════
from sklearn.cluster import KMeans

KM_DROP     = ["Monthly_Spend","Group","Risk_score"]
X_km        = scored_cln.drop(columns=KM_DROP, errors='ignore').copy().fillna(0).astype(float)
risk_vals   = scored_cln["Risk_score"].values

km_scaler   = StandardScaler()
X_km_scaled = km_scaler.fit_transform(X_km)

km         = KMeans(n_clusters=20, random_state=42, n_init=15)
raw_labels = km.fit_predict(X_km_scaled)

tmp = pd.DataFrame({"Cluster_Raw": raw_labels, "Risk_score": risk_vals})
cluster_risk_means = tmp.groupby("Cluster_Raw")["Risk_score"].mean().sort_values()

sorted_clusters = cluster_risk_means.index.tolist()
sorted_means    = cluster_risk_means.values
fusion_map      = {sorted_clusters[0]: 0}
group_id        = 0
FUSION_THRESHOLD = 5.0

for i in range(1, len(sorted_clusters)):
    if sorted_means[i] - sorted_means[i-1] < FUSION_THRESHOLD:
        fusion_map[sorted_clusters[i]] = group_id
    else:
        group_id += 1
        fusion_map[sorted_clusters[i]] = group_id

tmp["Cluster_Fused"] = tmp["Cluster_Raw"].map(fusion_map)
fused_risk   = tmp.groupby("Cluster_Fused")["Risk_score"].mean().sort_values()
reorder_map  = {old: new for new, old in enumerate(fused_risk.index)}
final_map    = {raw: reorder_map[fused] for raw, fused in fusion_map.items()}
n_clusters   = len(reorder_map)

labels_final = np.array([final_map[r] for r in raw_labels])
cluster_df   = pd.DataFrame(X_km.copy())
cluster_df["Cluster"]    = labels_final
cluster_df["Risk_score"] = risk_vals
cluster_profiles = cluster_df.groupby("Cluster").mean()

joblib.dump(km,                        "models/kmeans_model.joblib")
joblib.dump(km_scaler,                 "models/kmeans_scaler.joblib")
joblib.dump(final_map,                 "models/kmeans_fusion_map.joblib")
joblib.dump(X_km.columns.tolist(),     "models/kmeans_feature_cols.joblib")
joblib.dump(n_clusters,                "models/n_clusters.joblib")
joblib.dump(cluster_profiles,          "models/cluster_profiles.joblib")

print(f"✅ KMeans saved | {n_clusters} clusters after fusion")
print(f"\n🎉 All models saved! Files: {os.listdir('models')}")
