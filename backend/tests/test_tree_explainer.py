import os
import time
import joblib
import shap

from schemas.prediction_schema import LoanApplication
from services.model_service import prepare_input


# --------------------------------------------------
# Locate project and model
# --------------------------------------------------

BACKEND_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(BACKEND_DIR)

MODEL_PATH = os.path.join(
    PROJECT_ROOT,
    "models",
    "loan_default_model.joblib"
)


# --------------------------------------------------
# Load trained model
# --------------------------------------------------

print("Loading trained model...")

pipeline = joblib.load(MODEL_PATH)

imputer = pipeline.named_steps["imputer"]
model = pipeline.named_steps["model"]

print("Model loaded successfully.")


# --------------------------------------------------
# Create test applicant
# --------------------------------------------------

application = LoanApplication(
    RevolvingUtilizationOfUnsecuredLines=0.45,
    age=42,
    NumberOfTime30_59DaysPastDueNotWorse=1,
    DebtRatio=0.35,
    MonthlyIncome=6500,
    NumberOfOpenCreditLinesAndLoans=8,
    NumberOfTimes90DaysLate=0,
    NumberRealEstateLoansOrLines=1,
    NumberOfTime60_89DaysPastDueNotWorse=0,
    NumberOfDependents=2
)


# --------------------------------------------------
# Prepare input
# --------------------------------------------------

input_df = prepare_input(application)

transformed_input = imputer.transform(input_df)

print()
print("Original input features:", input_df.shape[1])
print("Transformed features:", transformed_input.shape[1])


# --------------------------------------------------
# Create TreeExplainer
# --------------------------------------------------

print()
print("Creating TreeExplainer...")

try:

    explainer = shap.TreeExplainer(model)

    print("TreeExplainer created successfully.")

except Exception as error:

    print()
    print("TreeExplainer could not be created.")
    print("Error type:", type(error).__name__)
    print("Error:", error)

    raise


# --------------------------------------------------
# Calculate SHAP values
# --------------------------------------------------

print()
print("Calculating SHAP values...")

start_time = time.perf_counter()

shap_values = explainer.shap_values(transformed_input)

end_time = time.perf_counter()

elapsed_time = end_time - start_time


# --------------------------------------------------
# Display results
# --------------------------------------------------

print()
print("=" * 60)
print("TREE SHAP TEST")
print("=" * 60)

print(f"Explanation time: {elapsed_time:.4f} seconds")

print("SHAP output type:", type(shap_values))

print(
    "SHAP output shape:",
    getattr(shap_values, "shape", "N/A")
)

print("=" * 60)



print()
print("TRANSFORMED FEATURE NAMES")
print("=" * 60)

feature_names = imputer.get_feature_names_out()

for index, name in enumerate(feature_names):
    print(f"{index}: {name}")

print()
print("SHAP VALUES")
print("=" * 60)

for index, value in enumerate(shap_values[0]):
    print(f"{feature_names[index]}: {value:.6f}")