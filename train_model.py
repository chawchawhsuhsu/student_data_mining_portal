import pandas as pd
import numpy as np
import joblib
from sklearn.model_selection import train_test_split
from sklearn.ensemble import ExtraTreesClassifier
from sklearn.preprocessing import OrdinalEncoder

# 1. Load Dataset
DATASET_FILE = "english_learning_dataset_2000.xlsx"
df = pd.read_excel(DATASET_FILE)

# 2. Define Target and Features (Ensure all 19 columns are included)
target_col = 'Observed Learning Progress Track'

# Drop target column to isolate all 19 predictor features
X = df.drop(columns=[target_col])
y = df[target_col]

# 3. Handle Categorical Columns using Ordinal Encoding (Prevents Feature Dilution)
categorical_cols = X.select_dtypes(include=['object', 'category']).columns
X_encoded = X.copy()

if len(categorical_cols) > 0:
    encoder = OrdinalEncoder(handle_unknown='use_encoded_value', unknown_value=-1)
    X_encoded[categorical_cols] = encoder.fit_transform(X[categorical_cols])

# 4. Save Clean Feature Names (Exactly 19 Features)
feature_names = list(X_encoded.columns)
joblib.dump(feature_names, "model_features.pkl")
print(f"Saved {len(feature_names)} features to model_features.pkl")

# 5. Train Model (ExtraTrees forces evaluation across ALL 19 features)
X_train, X_test, y_train, y_test = train_test_split(X_encoded, y, test_size=0.2, random_state=42)

model = ExtraTreesClassifier(
    n_estimators=100,
    max_depth=10,
    min_samples_leaf=2,
    random_state=42
)
model.fit(X_train, y_train)

# 6. Save Updated Model
joblib.dump(model, "decision_tree_model.pkl")
print("Successfully trained and saved updated model to decision_tree_model.pkl!")

# Diagnostic Output
importances = model.feature_importances_
fi_df = pd.DataFrame({'Feature': feature_names, 'Importance': importances}).sort_values('Importance', ascending=False)
print("\nFeature Importances Breakdown:")
print(fi_df)