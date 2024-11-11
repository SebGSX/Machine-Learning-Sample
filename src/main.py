# © 2024 Seb Garrioch. All rights reserved.
# Published under the MIT License.
import os
import pandas as pd

from src.config.config_manager import ConfigManager
from src.models import ModelCoreEducational, SpnnModel

if __name__ == "__main__": # pragma: no cover
    os.environ["KAGGLE_CONFIG_DIR"] = ".kaggle"
    config_manager = ConfigManager("config/config.json")
    config = config_manager.load_config()
    active_dataset: int = config["kaggle"]["active_dataset"]
    dataset_config: dict = config["kaggle"]["datasets"][active_dataset]
    feature_names = dataset_config["feature_names"]
    model = SpnnModel(ModelCoreEducational(), "../output/")
    model.setup_linear_regression_training(
        feature_names
        , dataset_config["label"]
        , 1e-8
        , 5
        , 100
        , 0.75
        , dataset_config["handle"]
        , dataset_config["file_name"]
        , dataset_config["competition_dataset"])
    model.train_linear_regression(True)

    y_hat = None

    if active_dataset == 0:
        y_hat = model.predict(pd.DataFrame({
            feature_names[0]: [1710, 1200, 2200],
            feature_names[1]: [7, 6, 8]
        }))

    if active_dataset == 1:
        y_hat = model.predict(pd.DataFrame({ feature_names[0]: [50, 120, 200] }))

    print(y_hat)
