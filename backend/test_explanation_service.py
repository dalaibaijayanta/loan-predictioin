from schemas.prediction_schema import LoanApplication

from services.model_service import (
    prepare_input
)

from services.explanation_service import (
    explain_prediction
)


# =========================================================
# TEST APPLICATION
# =========================================================

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


# =========================================================
# PREPARE INPUT
# =========================================================

input_df = prepare_input(
    application
)


# =========================================================
# GET EXPLANATION
# =========================================================

explanations = explain_prediction(
    input_df
)


# =========================================================
# PRINT RESULT
# =========================================================

print()
print("=" * 60)
print("TOP FEATURES AFFECTING THIS PREDICTION")
print("=" * 60)


for explanation in explanations:

    direction = explanation[
        "direction"
    ]

    symbol = (
        "↑"
        if direction == "increases_risk"
        else "↓"
    )

    print(
        f"{symbol} "
        f"{explanation['feature']}"
        f" | SHAP: "
        f"{explanation['impact']}"
    )