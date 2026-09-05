from financial_ai.agents.state import FinancialAgentState

from financial_ai.config.settings import settings

from financial_ai.ml.inference import (
    predict_financial_outlook,
)

from financial_ai.ml.model_config import (
    DIRECTION_APPROVED_PARAMS,
    VOLATILITY_APPROVED_PARAMS,
)

import pandas as pd


def ml_agent_node(
    state: FinancialAgentState,
) -> FinancialAgentState:

    if not state.get("use_ml", False):
        return state

    try:

        ticker = state["ticker"].upper()

        dataset_path = (
            settings.FEATURE_DATA_DIR
            / "training"
            / f"{ticker}_ml_features.parquet"
        )

        dataset = pd.read_parquet(
            dataset_path
        )

        result = predict_financial_outlook(
            dataset=dataset,
            inference_date=state["as_of_date"],
            direction_params=DIRECTION_APPROVED_PARAMS,
            volatility_params=VOLATILITY_APPROVED_PARAMS,
        )

        return {
            **state,
            "ml_result": result,
            "ml_error": None,
        }

    except Exception as exc:

        return {
            **state,
            "ml_result": None,
            "ml_error": str(exc),
        }