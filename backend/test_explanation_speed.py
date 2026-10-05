import time

from schemas.prediction_schema import LoanApplication

from services.model_service import (
    prepare_input
)

from services.explanation_service import (
    explain_prediction
)


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


input_df = prepare_input(
    application
)


start_time = time.perf_counter()

explanations = explain_prediction(
    input_df
)

end_time = time.perf_counter()


elapsed_time = end_time - start_time


print()
print("=" * 60)
print("SHAP EXPLANATION SPEED")
print("=" * 60)

print(
    f"Explanation time: "
    f"{elapsed_time:.2f} seconds"
)

print()
print("Explanation:")

for item in explanations:

    print(
        f"{item['feature']} | "
        f"{item['impact']} | "
        f"{item['direction']}"
    )