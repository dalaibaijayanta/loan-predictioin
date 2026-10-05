from fastapi import APIRouter

from schemas.prediction_schema import LoanApplication

from services.model_service import (
    prepare_input,
    predict
)

from services.explanation_service import (
    explain_prediction
)


router = APIRouter()


@router.post("/predict")
def predict_loan(application: LoanApplication):

    # -----------------------------------------------------
    # 1. Convert validated application into DataFrame
    # -----------------------------------------------------

    input_df = prepare_input(application)


    # -----------------------------------------------------
    # 2. Get ML prediction
    # -----------------------------------------------------

    prediction_result = predict(
        application
    )


    # -----------------------------------------------------
    # 3. Generate SHAP explanation
    # -----------------------------------------------------

    explanations = explain_prediction(
        input_df
    )


    # -----------------------------------------------------
    # 4. Combine results
    # -----------------------------------------------------

    return {
        **prediction_result,
        "explanations": explanations
    }