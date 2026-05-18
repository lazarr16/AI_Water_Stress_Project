import pandas as pd

from sklearn.model_selection import (
    train_test_split,
    cross_val_score,
    StratifiedKFold
)

from sklearn.metrics import (
    classification_report,
    accuracy_score,
    confusion_matrix
)

from sklearn.preprocessing import (
    LabelEncoder
)

from xgboost import XGBClassifier

# -------------------------
# Load dataset
# -------------------------

df = pd.read_csv(
    "data/processed/final_model_dataset.csv"
)

print("Dataset loaded!")

# -------------------------
# Features & Target
# -------------------------

feature_columns = [

    "avg_temperature",
    "avg_humidity",
    "precipitation",

    "data_center_count",
    "estimated_mw_capacity",
    "hyperscaler_presence",

    "cooling_severity_index_v2",
    "ai_infrastructure_pressure_score"
]

X = df[feature_columns]

y = df["water_stress_risk"]

# -------------------------
# Encode target
# -------------------------

encoder = LabelEncoder()

y_encoded = encoder.fit_transform(y)

print("\nClasses:")
print(encoder.classes_)

# -------------------------
# Train/Test Split
# -------------------------

X_train, X_test, y_train, y_test = (
    train_test_split(
        X,
        y_encoded,
        test_size=0.2,
        random_state=42,
        stratify=y_encoded
    )
)

# -------------------------
# XGBoost Model
# -------------------------

model = XGBClassifier(
    n_estimators=200,
    max_depth=5,
    learning_rate=0.05,
    random_state=42,
    eval_metric="mlogloss"
)

model.fit(
    X_train,
    y_train
)

print("\nModel trained!")

# -------------------------
# Predictions
# -------------------------

y_pred = model.predict(X_test)

# -------------------------
# Evaluation
# -------------------------

accuracy = accuracy_score(
    y_test,
    y_pred
)

print("\nAccuracy:")
print(round(accuracy, 4))

print("\nClassification Report:")
print(
    classification_report(
        y_test,
        y_pred,
        target_names=
        encoder.classes_
    )
)

print("\nConfusion Matrix:")
print(
    confusion_matrix(
        y_test,
        y_pred
    )
)

# -------------------------
# Cross Validation
# -------------------------

cv = StratifiedKFold(
    n_splits=5,
    shuffle=True,
    random_state=42
)

cv_scores = cross_val_score(
    model,
    X,
    y_encoded,
    cv=cv,
    scoring="accuracy"
)

print("\nCross Validation Scores:")
print(cv_scores)

print("\nAverage CV Accuracy:")
print(round(cv_scores.mean(), 4))

print("\nCV Standard Deviation:")
print(round(cv_scores.std(), 4))

# -------------------------
# Feature Importance
# -------------------------

importance_df = pd.DataFrame({
    "feature":
    feature_columns,

    "importance":
    model.feature_importances_
})

importance_df = (
    importance_df
    .sort_values(
        by="importance",
        ascending=False
    )
)

print("\nFeature Importance:")
print(importance_df)