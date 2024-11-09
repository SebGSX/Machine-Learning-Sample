# © 2024 Seb Garrioch. All rights reserved.
# Published under the MIT License.
import os
import pandas as pd

from src.config.config_manager import ConfigManager
from src.models import SpnnModel

if __name__ == "__main__": # pragma: no cover
    os.environ["KAGGLE_CONFIG_DIR"] = ".kaggle"
    config_manager = ConfigManager("config/config.json")
    config = config_manager.load_config()
    active_dataset: int = config["kaggle"]["active_dataset"]
    dataset_config: dict = config["kaggle"]["datasets"][active_dataset]
    feature_name = dataset_config["feature_names"][0]
    model = SpnnModel("../output/")
    model.setup_linear_regression_training(
        [ feature_name ]
        , dataset_config["label"]
        , 1e-8
        , 5
        , 100
        , 0.75
        , dataset_config["handle"]
        , dataset_config["file_name"]
        , dataset_config["competition_dataset"])
    model.train_linear_regression(True)
    y_hat = model.predict(pd.DataFrame({ feature_name: [50, 120, 200] }))
    print(y_hat)
