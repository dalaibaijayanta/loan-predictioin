from schemas.prediction_schema import LoanApplication
from services.model_service import predict


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


result = predict(application)

print()
print("=" * 50)
print("MODEL PREDICTION")
print("=" * 50)

print(result)