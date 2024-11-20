import numpy as np

from pipeline import price_pipe
from config.core import config
from sklearn.model_selection import train_test_split
from processing.data_manager import load_dataset, save_pipeline

# Train the model
def run_training() -> None:
    # read training data
    data = load_dataset(file_name=config.app_config.training_data_file)

    # divide train and test set
    X_train, X_test, y_train, y_test = train_test_split(
        data[config.model_config.features],
        data[config.model_config.target],
        test_size=config.model_config.test_size,
        random_state=config.model_config.random_state
    )
    y_train = np.log(y_train)

    # fit model
    price_pipe.fit(X_train, y_train)

    # persist trained model
    save_pipeline(pipeline_to_persist=price_pipe)

if __name__ == '__main__':
    run_training()