import pandas as pd

from sklearn.model_selection import train_test_split, RandomizedSearchCV
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.ensemble import HistGradientBoostingClassifier

from sklearn.metrics import (
    roc_auc_score,
    recall_score,
    precision_score,
    f1_score
)

from ml.preprocessing.cleaning import clean_raw_data


# ---------------------------------------------------------
# Configuration
# ---------------------------------------------------------

DATA_PATH = "data/cs-training.csv"
RANDOM_STATE = 42


# ---------------------------------------------------------
# Load and clean data
# ---------------------------------------------------------

print("=" * 70)
print("LOADING DATA")
print("=" * 70)

df = pd.read_csv(DATA_PATH)

print(f"Original shape: {df.shape}")

df = clean_raw_data(df)

print(f"Shape after cleaning: {df.shape}")


# ---------------------------------------------------------
# Features and target
# ---------------------------------------------------------

X = df.drop(columns=["SeriousDlqin2yrs"])
y = df["SeriousDlqin2yrs"]


# ---------------------------------------------------------
# Train / Validation / Test split
# ---------------------------------------------------------

X_train, X_temp, y_train, y_temp = train_test_split(
    X,
    y,
    test_size=0.20,
    stratify=y,
    random_state=RANDOM_STATE
)

X_val, X_test, y_val, y_test = train_test_split(
    X_temp,
    y_temp,
    test_size=0.50,
    stratify=y_temp,
    random_state=RANDOM_STATE
)


# ---------------------------------------------------------
# Preprocessing + model pipeline
# ---------------------------------------------------------

pipeline = Pipeline(
    steps=[
        (
            "imputer",
            SimpleImputer(
                strategy="median",
                add_indicator=True
            )
        ),

        (
            "classifier",
            HistGradientBoostingClassifier(
                random_state=RANDOM_STATE
            )
        )
    ]
)


# ---------------------------------------------------------
# Hyperparameter search space
# ---------------------------------------------------------

param_distributions = {
    "classifier__max_iter": [
        200,
        300,
        400,
        500
    ],

    "classifier__learning_rate": [
        0.02,
        0.05,
        0.08,
        0.10
    ],

    "classifier__max_leaf_nodes": [
        15,
        31,
        63
    ],

    "classifier__min_samples_leaf": [
        10,
        20,
        30,
        50
    ],

    "classifier__l2_regularization": [
        0.0,
        0.1,
        1.0,
        5.0
    ]
}


# ---------------------------------------------------------
# Randomized Search
# ---------------------------------------------------------

print("\n" + "=" * 70)
print("STARTING HYPERPARAMETER TUNING")
print("=" * 70)

print("Scoring metric: ROC-AUC")
print("Cross-validation folds: 3")
print("Random parameter combinations: 15")
print("\nThis may take some time...")


search = RandomizedSearchCV(
    estimator=pipeline,
    param_distributions=param_distributions,
    n_iter=15,
    scoring="roc_auc",
    cv=3,
    random_state=RANDOM_STATE,
    n_jobs=-1,
    verbose=1
)


# ---------------------------------------------------------
# Fit search ONLY on training data
# ---------------------------------------------------------

search.fit(X_train, y_train)


# ---------------------------------------------------------
# Best result
# ---------------------------------------------------------

print("\n" + "=" * 70)
print("BEST MODEL")
print("=" * 70)

print(f"Best CV ROC-AUC: {search.best_score_:.4f}")

print("\nBest parameters:")

for parameter, value in search.best_params_.items():
    print(f"{parameter}: {value}")


# ---------------------------------------------------------
# Validation evaluation
# ---------------------------------------------------------

best_model = search.best_estimator_

y_val_probability = best_model.predict_proba(X_val)[:, 1]


# ROC-AUC
roc_auc = roc_auc_score(
    y_val,
    y_val_probability
)


# ---------------------------------------------------------
# Threshold = 0.20
# ---------------------------------------------------------

threshold = 0.20

y_val_pred = (
    y_val_probability >= threshold
).astype(int)


recall = recall_score(
    y_val,
    y_val_pred
)

precision = precision_score(
    y_val,
    y_val_pred
)

f1 = f1_score(
    y_val,
    y_val_pred
)


# ---------------------------------------------------------
# Results
# ---------------------------------------------------------

print("\n" + "=" * 70)
print("VALIDATION RESULTS")
print("=" * 70)
print(f"Threshold : {threshold:.2f}")
print(f"Recall    : {recall:.4f}")
print(f"Precision : {precision:.4f}")
print(f"F1 Score  : {f1:.4f}")
print(f"ROC-AUC   : {roc_auc:.4f}")


# ---------------------------------------------------------
# Compare with previous model
# ---------------------------------------------------------

print("\n" + "=" * 70)
print("PREVIOUS MODEL")
print("=" * 70)

print("ROC-AUC   : 0.8670")
print("Recall    : 0.5279")
print("Precision : 0.3951")
print("F1 Score  : 0.4519")


print("\n" + "=" * 70)
print("TEST SET")
print("=" * 70)

print("Test set was NOT used during tuning or evaluation.")