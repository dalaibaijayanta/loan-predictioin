import os
import joblib
import pandas as pd


# =========================================================
# FIND PROJECT ROOT
# =========================================================

CURRENT_DIR = os.path.dirname(
    os.path.abspath(__file__)
)

PROJECT_ROOT = os.path.abspath(
    os.path.join(CURRENT_DIR, "..", "..")
)


# =========================================================
# MODEL FILES
# =========================================================

MODEL_PATH = os.path.join(
    PROJECT_ROOT,
    "models",
    "loan_default_model.joblib"
)

THRESHOLD_PATH = os.path.join(
    PROJECT_ROOT,
    "models",
    "threshold.txt"
)


# =========================================================
# LOAD MODEL
# =========================================================

print("Loading ML model...")

model = joblib.load(MODEL_PATH)

with open(THRESHOLD_PATH, "r") as file:
    THRESHOLD = float(
        file.read().strip()
    )

print("ML model loaded successfully.")
print(f"Decision threshold: {THRESHOLD}")


# =========================================================
# CONVERT APPLICATION TO DATAFRAME
# =========================================================

def prepare_input(application):

    data = {
        "RevolvingUtilizationOfUnsecuredLines":
            application.RevolvingUtilizationOfUnsecuredLines,

        "age":
            application.age,

        "NumberOfTime30-59DaysPastDueNotWorse":
            application.NumberOfTime30_59DaysPastDueNotWorse,

        "DebtRatio":
            application.DebtRatio,

        "MonthlyIncome":
            application.MonthlyIncome,

        "NumberOfOpenCreditLinesAndLoans":
            application.NumberOfOpenCreditLinesAndLoans,

        "NumberOfTimes90DaysLate":
            application.NumberOfTimes90DaysLate,

        "NumberRealEstateLoansOrLines":
            application.NumberRealEstateLoansOrLines,

        "NumberOfTime60-89DaysPastDueNotWorse":
            application.NumberOfTime60_89DaysPastDueNotWorse,

        "NumberOfDependents":
            application.NumberOfDependents
    }

    return pd.DataFrame([data])


# =========================================================
# MAKE PREDICTION
# =========================================================

def predict(application):

    input_df = prepare_input(application)

    probability = model.predict_proba(
        input_df
    )[0][1]

    if probability >= THRESHOLD:
        risk_level = "HIGH RISK"
    else:
        risk_level = "LOW RISK"

    risk_score = round(
        probability * 100
    )

    return {
        "default_probability": round(
            probability * 100,
            2
        ),
        "risk_score": risk_score,
        "risk_level": risk_level
    }