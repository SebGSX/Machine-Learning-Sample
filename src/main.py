# © 2024 Seb Garrioch. All rights reserved.
# Published under the MIT License.
import pandas as pd

from src.models import SpnnModel

if __name__ == "__main__": # pragma: no cover
    model = SpnnModel("../output/")
    model.setup_linear_regression_training(
        ["TV"]
        , "Sales"
        , 1e-8
        , 5
        , 100
        , 0.75
        , "devzohaib/tvmarketingcsv"
        , "tvmarketing.csv")
    model.train_linear_regression(True)
    y_hat = model.predict(pd.DataFrame({ "TV": [50, 120, 200] }))
    print(y_hat)
