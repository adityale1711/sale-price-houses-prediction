import pytest
import pandas as pd

from app.main import app
from typing import Generator
from fastapi.testclient import TestClient
from sale_price_house_prediction_model.config.core import config
from sale_price_house_prediction_model.processing.data_manager import load_dataset


@pytest.fixture(scope='module')
def test_data() -> pd.DataFrame:
    return load_dataset(file_name=config.app_cnf.test_data_file)


@pytest.fixture()
def client() -> Generator:
    with TestClient(app) as _client:
        yield _client
        app.dependency_overrides = {}
