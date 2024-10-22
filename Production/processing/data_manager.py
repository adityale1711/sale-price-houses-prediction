import pandas as pd

from pathlib import Path
from Production.config.core import config, DATASET_DIR

def load_dataset(*, file_name: str) -> pd.DataFrame:
    dataframe = pd.read_csv(Path(f'{DATASET_DIR}/{file_name}'))
    dataframe['MSSubClass'] = dataframe['MSSubClass'].astype('O')

    # rename variables begining with numbers to avoid syntax errors later
    transformed = dataframe.rename(columns=config.model_config.variables_to_rename)

    return transformed