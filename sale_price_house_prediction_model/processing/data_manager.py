import joblib
import typing as t
import pandas as pd

from pathlib import Path
from Production import __version__ as _version
from sklearn.pipeline import Pipeline
from sale_price_house_prediction_model.config.core import config, DATASET_DIR, TRAINED_MODEL_DIR

def load_dataset(*, file_name: str) -> pd.DataFrame:
    dataframe = pd.read_csv(Path(f'{DATASET_DIR}/{file_name}'))
    dataframe['MSSubClass'] = dataframe['MSSubClass'].astype('O')

    # rename variables begining with numbers to avoid syntax errors later
    transformed = dataframe.rename(columns=config.model_config.variables_to_rename)

    return transformed

# Remove old model pipeline
def remove_old_pipeline(*, files_to_keep: t.List[str]) -> None:
    dont_delete = files_to_keep + ['__init__.py']
    for model_file in TRAINED_MODEL_DIR.iterdir():
        if model_file.name not in dont_delete:
            model_file.unlink()

# Persist the pipeline
def save_pipeline(*, pipeline_to_persist: Pipeline) -> None:
    # Prepare versioned save file name
    save_file_name = f'{config.app_config.pipeline_save_file}{_version}.pkl'
    save_path = TRAINED_MODEL_DIR / save_file_name

    remove_old_pipeline(files_to_keep=[save_file_name])
    joblib.dump(pipeline_to_persist, save_path)

# Load a persisted pipeline
def load_pipeline(*, file_name: str) -> Pipeline:
    file_path = TRAINED_MODEL_DIR /file_name
    trained_model = joblib.load(filename=file_path)

    return trained_model