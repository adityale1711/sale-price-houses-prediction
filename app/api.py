import json

import numpy as np
import pandas as pd
from matplotlib.font_manager import json_load

from app import __version__, schemas
from typing import Any
from loguru import logger
from fastapi import APIRouter, HTTPException
from app.config import settings
from fastapi.encoders import jsonable_encoder
from sale_price_house_prediction_model import __version__ as model_version
from sale_price_house_prediction_model.predict import make_prediction

api_router = APIRouter()


# Root get
@api_router.get('/health', response_model=schemas.Health, status_code=200)
def health() -> dict:
    health = schemas.Health(
        name=settings.PROJECT_NAME,
        api_version=__version__,
        model_version=model_version
    )

    return health.dict()


# Make House Price Prediction with the TID sale house price prediction
@api_router.post('/predict', response_model=schemas.PredictionResults, status_code=200)
async def predict(input_data: schemas.MultipleHouseDataInputs) -> Any:
    input_df = pd.DataFrame(jsonable_encoder(input_data.inputs))

    # 'make_prediction' function to be async and using await here
    logger.info(f'Making prediction on inputs: {input_data.inputs}')
    results = make_prediction(input_data=input_df.replace({np.nan: None}))

    if results['errors'] is not None:
        logger.warning(f"Prediction validation error: {results.get('errors')}")
        raise HTTPException(status_code=400, detail=json.loads(results['errors']))

    logger.info(f"Prediction results: {results.get('prediction')}")

    return results