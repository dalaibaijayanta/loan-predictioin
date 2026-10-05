import os
import joblib
import shap


# --------------------------------------------------
# Locate project
# --------------------------------------------------

CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))

BACKEND_DIR = os.path.dirname(CURRENT_DIR)

PROJECT_ROOT = os.path.dirname(BACKEND_DIR)

MODEL_PATH = os.path.join(
    PROJECT_ROOT,
    "models",
    "loan_default_model.joblib"
)


# --------------------------------------------------
# Feature names used by the application
# --------------------------------------------------

DISPLAY_NAMES = {
    "RevolvingUtilizationOfUnsecuredLines":
        "Revolving Credit Utilization",

    "age":
        "Age",

    "NumberOfTime30-59DaysPastDueNotWorse":
        "30-59 Days Past Due",

    "DebtRatio":
        "Debt Ratio",

    "MonthlyIncome":
        "Monthly Income",

    "NumberOfOpenCreditLinesAndLoans":
        "Open Credit Lines and Loans",

    "NumberOfTimes90DaysLate":
        "90+ Days Late",

    "NumberRealEstateLoansOrLines":
        "Real Estate Loans or Lines",

    "NumberOfTime60-89DaysPastDueNotWorse":
        "60-89 Days Past Due",

    "NumberOfDependents":
        "Number of Dependents",

    "missingindicator_age":
        "Age Missing",

    "missingindicator_NumberOfTime30-59DaysPastDueNotWorse":
        "30-59 Days Past Due Missing",

    "missingindicator_MonthlyIncome":
        "Monthly Income Missing",

    "missingindicator_NumberOfTimes90DaysLate":
        "90+ Days Late Missing",

    "missingindicator_NumberOfTime60-89DaysPastDueNotWorse":
        "60-89 Days Past Due Missing",

    "missingindicator_NumberOfDependents":
        "Number of Dependents Missing",
}


# --------------------------------------------------
# Load model once when server starts
# --------------------------------------------------

print("Loading model for explainability...")

pipeline = joblib.load(MODEL_PATH)

imputer = pipeline.named_steps["imputer"]

model = pipeline.named_steps["model"]

print("Creating TreeExplainer...")

explainer = shap.TreeExplainer(model)

print("TreeExplainer ready.")


# --------------------------------------------------
# Explain prediction
# --------------------------------------------------

def explain_prediction(input_df, top_n=5):

    # Transform input using the SAME imputer
    # that was used during model training.
    transformed_input = imputer.transform(input_df)

    # Calculate SHAP values
    shap_values = explainer.shap_values(transformed_input)

    # Get feature names after imputation
    feature_names = imputer.get_feature_names_out()

    # First/only applicant
    values = shap_values[0]

    explanations = []

    for feature_name, impact in zip(feature_names, values):

        feature_name = str(feature_name)

        impact = float(impact)

        # Ignore extremely tiny contributions
        if abs(impact) < 0.000001:
            continue

        display_name = DISPLAY_NAMES.get(
            feature_name,
            feature_name
        )

        if impact > 0:
            direction = "increases_risk"
        else:
            direction = "decreases_risk"

        explanations.append(
            {
                "feature": display_name,
                "impact": round(impact, 5),
                "direction": direction
            }
        )

    # Sort by absolute SHAP impact
    explanations.sort(
        key=lambda item: abs(item["impact"]),
        reverse=True
    )

    # Return only the strongest explanations
    return explanations[:top_n]