from schemas.prediction_schema import LoanApplication


# Valid application
valid_application = LoanApplication(
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

print("VALID APPLICATION")
print(valid_application)


# Missing values
application_with_missing_values = LoanApplication(
    RevolvingUtilizationOfUnsecuredLines=0.45,
    age=42,
    NumberOfTime30_59DaysPastDueNotWorse=1,
    DebtRatio=0.35,
    MonthlyIncome=None,
    NumberOfOpenCreditLinesAndLoans=8,
    NumberOfTimes90DaysLate=0,
    NumberRealEstateLoansOrLines=1,
    NumberOfTime60_89DaysPastDueNotWorse=0,
    NumberOfDependents=None
)

print()
print("APPLICATION WITH MISSING VALUES")
print(application_with_missing_values)


# Invalid application
try:

    invalid_application = LoanApplication(
        RevolvingUtilizationOfUnsecuredLines=0.45,
        age=-10,
        NumberOfTime30_59DaysPastDueNotWorse=1,
        DebtRatio=0.35,
        MonthlyIncome=6500,
        NumberOfOpenCreditLinesAndLoans=8,
        NumberOfTimes90DaysLate=0,
        NumberRealEstateLoansOrLines=1,
        NumberOfTime60_89DaysPastDueNotWorse=0,
        NumberOfDependents=2
    )

except Exception as error:

    print()
    print("INVALID APPLICATION REJECTED")
    print(error)