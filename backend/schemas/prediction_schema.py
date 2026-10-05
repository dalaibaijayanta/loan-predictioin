from pydantic import BaseModel, Field


class LoanApplication(BaseModel):

    RevolvingUtilizationOfUnsecuredLines: float = Field(
        ge=0,
        description="Revolving unsecured credit utilization"
    )

    age: int = Field(
        ge=18,
        le=120,
        description="Applicant age"
    )

    NumberOfTime30_59DaysPastDueNotWorse: int = Field(
        ge=0,
        description="Number of times 30-59 days past due"
    )

    DebtRatio: float = Field(
        ge=0,
        description="Debt ratio"
    )

    MonthlyIncome: float | None = Field(
        default=None,
        ge=0,
        description="Monthly income"
    )

    NumberOfOpenCreditLinesAndLoans: int = Field(
        ge=0,
        description="Number of open credit lines and loans"
    )

    NumberOfTimes90DaysLate: int = Field(
        ge=0,
        description="Number of times 90+ days late"
    )

    NumberRealEstateLoansOrLines: int = Field(
        ge=0,
        description="Number of real estate loans or lines"
    )

    NumberOfTime60_89DaysPastDueNotWorse: int = Field(
        ge=0,
        description="Number of times 60-89 days past due"
    )

    NumberOfDependents: int | None = Field(
        default=None,
        ge=0,
        description="Number of dependents"
    )