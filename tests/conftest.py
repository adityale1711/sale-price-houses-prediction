import pytest

from sale_price_house_prediction_model.config.core import config
from sale_price_house_prediction_model.processing.data_manager import load_dataset

@pytest.fixture()
def sample_input_data():
    return load_dataset(file_name=config.app_cnf.test_data_file)