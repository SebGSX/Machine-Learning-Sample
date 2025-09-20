# © 2025 Seb Garrioch. All rights reserved.
# Published under the MIT License.
import kagglehub
import pandas as pd
import os


class DataSetManager:
    """Manages a list of datasets."""

    def __init__(self):
        """Initialises the dataset manager."""
        self.datasets: dict = {}

    def get_datasets(self) -> dict:
        """Gets the dictionary of datasets.
        :return: The dictionary of datasets.
        """
        return self.datasets

    def get_dataset(self, handle: str) -> pd.DataFrame:
        """Gets a dataset from the dictionary of datasets using its Kaggle handle.
        :return: The dataset.
        """
        return self.datasets[handle]

    def get_dataset_count(self) -> int:
        """Gets the number of datasets in the list.
        :return: The number of datasets.
        """
        return len(self.datasets)

    def acquire_kaggle_dataset(
            self
            , handle: str
            , file_name: str
            , competition: bool = False
            , force_download: bool = False): # pragma: no cover
        """Acquires a new dataset from Kaggle.
        :param handle: The dataset's Kaggle handle.
        :param file_name: The file name of the dataset, including the extension.
        :param competition: Indicates whether the dataset is part of a competition.
        :param force_download: Indicates whether to force the download of the dataset even if it is already cached.
        """
        try:
            if competition:
                path = kagglehub.competition_download(handle, force_download=force_download)
            else:
                path = kagglehub.dataset_download(handle, force_download=force_download)
            full_path = os.path.abspath(os.path.join(path, file_name))
            dataset = pd.read_csv(full_path)
            self.add_dataset(handle, dataset)
        except Exception as e:
            print(f"\033[91mAn error occurred while acquiring the dataset: {e}")
            print("Please check that:")
            print("\tthe Kaggle handle is correct; and")
            print("\tthe file name must include the extension.\033[0m")
            raise

    def add_dataset(self, handle:str, dataset: pd.DataFrame):
        """Adds a dataset to the list of datasets.
        :param handle: The dataset's Kaggle handle.
        :param dataset: The dataset to add.
        """
        self.datasets[handle] = dataset

    def clear_datasets(self):
        """Clears the dictionary of datasets."""
        self.datasets.clear()

    def remove_dataset(self, handle: str):
        """Removes a dataset from the dictionary of datasets.
        :param handle: The Kaggle handle of the dataset to remove.
        """
        if handle in self.datasets:
            self.datasets.pop(handle)
