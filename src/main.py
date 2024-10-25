# © 2024 Seb Garrioch. All rights reserved.
# Published under the MIT License.
from src.data import DataSetManager

if __name__ == "__main__":
    manager: DataSetManager = DataSetManager()
    manager.acquire_kaggle_dataset("devzohaib/tvmarketingcsv", "tvmarketing.csv")
    dataset = manager.get_dataset(0)
