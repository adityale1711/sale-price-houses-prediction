import numpy as np
import typing as t
import pandas as pd

from Production import __version__ as _version
from Production.config.core import config
from Production.processing.validation import validate_input
from Production.processing.data_manager import load_pipeline

pipeline_file_name = f'{config.app_config.pipeline_save_file}{_version}.pkl'
_price_pipe = load_pipeline(file_name=pipeline_file_name)

# Make prediction using a saved model pipeline
def make_prediction(*, input_data: t.Union[pd.DataFrame, dict]) -> dict:
    data = pd.DataFrame(input_data)
    validated_data, errors = validate_input(input_data=data)
    results = {
        'predictions': None,
        'version': _version,
        'errors':errors
    }

    if not errors:
        predictions = _price_pipe.predict(X=validated_data[config.model_config.features])
        results = {
            'predictions': [np.exp(pred) for pred in predictions],
            'version': _version,
            'errors': errors
        }

    return results