import Production

from typing import Dict, List, Sequence, Optional
from pathlib import Path
from pydantic import BaseModel
from strictyaml import YAML, load

PACKAGE_ROOT = Path(Production.__file__).resolve().parent
DATASET_DIR = PACKAGE_ROOT / 'datasets'
CONFIG_FILE_PATH = PACKAGE_ROOT / 'config.yml'

# All configuration that relevant to model training and feature engineering
class ModelConfig(BaseModel):
    variables_to_rename: Dict

# Master config object
class Config(BaseModel):
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
            return  parsed_config

    raise OSError(f'Did not find config file at path: {cfg_path}')

# Run validation on config values
def create_and_validate_config(parsed_config: YAML = None) -> Config:
    if parsed_config is None:
        parsed_config = fetch_config_from_yaml()

    # Specify the data attribute from the strictyaml YAML type
    _config = Config(model_config=ModelConfig(**parsed_config.data))

    return _config

config = create_and_validate_config()