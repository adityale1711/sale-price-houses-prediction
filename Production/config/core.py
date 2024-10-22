import Production

from typing import Dict, List, Sequence, Optional
from pathlib import Path
from pydantic import BaseModel
from strictyaml import YAML, load

PACKAGE_ROOT = Path(Production.__file__).resolve().parent
DATASET_DIR = PACKAGE_ROOT / 'datasets'
CONFIG_FILE_PATH = PACKAGE_ROOT / 'config.yml'
TRAINED_MODEL_DIR = PACKAGE_ROOT / 'trained_models'

# Application-Level config
class AppConfig(BaseModel):
    package_name: str
    training_data_file: str
    pipeline_save_file: str

# All configuration that relevant to model training and feature engineering
class ModelConfig(BaseModel):
    alpha: float
    target: str
    ref_var: str
    features: List[str]
    test_size: float
    qual_vars: List[str]
    finish_vars: List[str]
    garage_vars: List[str]
    random_state: int
    qual_mappings: Dict[str, int]
    exposure_vars: List[str]
    temporal_vars: List[str]
    binarize_vars: Sequence[str]
    garage_mappings: Dict[str, int]
    finish_mappings: Dict[str, int]
    categorical_vars: Sequence[str]
    exposure_mappings: Dict[str, int]
    variables_to_rename: Dict
    numerical_log_vars: Sequence[str]
    numerical_vars_with_na: List[str]
    categorical_vars_with_na_missing: List[str]
    categorical_vars_with_na_frequent: List[str]

# Master config object
class Config(BaseModel):
    app_config: AppConfig
    model_config: ModelConfig

# Locate the configuration file
def find_config_file() -> Path:
    if CONFIG_FILE_PATH.is_file():
        return CONFIG_FILE_PATH

    raise Exception(f'Config not found at {CONFIG_FILE_PATH!r}')

# Parse YAML containing the package configuration
def fetch_config_from_yaml(cfg_path: Optional[Path] = None) -> YAML:
    if not cfg_path:
        cfg_path = find_config_file()

    if cfg_path:
        with open(cfg_path, 'r') as conf_file:
            parsed_config = load(conf_file.read())
            return parsed_config

    raise OSError(f'Did not find config file at path: {cfg_path}')

# Run validation on config values
def create_and_validate_config(parsed_config: YAML = None) -> Config:
    if parsed_config is None:
        parsed_config = fetch_config_from_yaml()

    # Specify the data attribute from the strictyaml YAML type
    _config = Config(app_config=AppConfig(**parsed_config.data), model_config=ModelConfig(**parsed_config.data))

    return _config

config = create_and_validate_config()