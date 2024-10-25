# © 2024 Seb Garrioch. All rights reserved.
# Published under the MIT License.
import kagglehub
import pandas as pd
import os


class DataSetManager:
    """Manages a list of datasets."""

    def __init__(self):
        """Initializes the dataset manager."""
        self.datasets: list = []

    def acquire_kaggle_dataset(self, handle: str, file_name: str, force_download: bool = False): # pragma: no cover
        """Acquires a new dataset from Kaggle.
        :param handle: The dataset's Kaggle handle.
        :param file_name: The file name of the dataset, including the extension.
        :param force_download: Indicates whether to force the download of the dataset even if it is already cached.
        """
        # Cannot unit test this code without making an actual Kaggle API call. It is therefore excluded.
        try:
            path = kagglehub.dataset_download(handle, force_download=force_download)
            full_path = os.path.abspath(os.path.join(path, file_name))
            dataset = pd.read_csv(full_path)
            self.add_dataset(dataset)
        except Exception as e:
            print(f"\033[91mAn error occurred while acquiring the dataset: {e}")
            print("Please check that:")
            print("\tthe Kaggle handle is correct; and")
            print("\tthe file name must include the extension.\033[0m")
            raise

    def add_dataset(self, dataset: pd.DataFrame):
        """Adds a dataset to the list of datasets.
        :param dataset: The dataset to add.
        """
        self.datasets.append(dataset)

    def get_datasets(self) -> list:
        """Gets the list of datasets.
        :return: The list of datasets.
        """
        return self.datasets

    def get_dataset(self, index: int) -> pd.DataFrame:
        """Gets a dataset from the list of datasets.
        :param index: The index of the dataset to retrieve.
        :return: The dataset.
        """
        return self.datasets[index]

    def get_dataset_count(self) -> int:
        """Gets the number of datasets in the list.
        :return: The number of datasets.
        """
        return len(self.datasets)

    def remove_dataset(self, index):
        """Removes a dataset from the list of datasets.
        :param index: The index of the dataset to remove.
        """
        self.datasets.pop(index)

    def clear_datasets(self):
        """Clears the list of datasets."""
        self.datasets.clear()
