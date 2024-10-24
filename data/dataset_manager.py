# © 2024 Seb Garrioch. All rights reserved.
# Published under the MIT License.
import kagglehub
import pandas as pd
import os


class DataSetManager:
    """Manages a list of data sets."""

    def __init__(self):
        """Initializes the data set manager."""
        self.datasets: list = []

    def acquire_kaggle_dataset(self, handle: str, file_name: str): # pragma: no cover
        """Acquires a new dataset from Kaggle.
        :param handle: The dataset's Kaggle handle.
        :param file_name: The file name of the dataset, including the extension.
        """
        # While I don't like tightly coupling the data set manager to KaggleHub or Pandas, it is necessary. At the
        # time of writing, Python doesn't support interfaces. While I could use duck typing or an abstract base class,
        # doing so would overcomplicate the code by burying the tight coupling under layers of indirection.
        try:
            path = kagglehub.dataset_download(handle, force_download=True)
            full_path = os.path.abspath(os.path.join(path, file_name))
            pd.read_csv(full_path)
            self.add_dataset(pd)
        except Exception as e:
            print(f"\033[91mAn error occurred while acquiring the dataset: {e}")
            print("Please check that:")
            print("\tthe Kaggle handle is correct; and")
            print("\tthe file name must include the extension.\033[0m")
            raise

    def add_dataset(self, dataset: object):
        """Adds a data set to the list of data sets.
        :param dataset: The data set to add.
        """
        self.datasets.append(dataset)

    def get_datasets(self) -> list:
        """Gets the list of data sets.
        :return: The list of data sets.
        """
        return self.datasets

    def get_dataset(self, index: int) -> object:
        """Gets a data set from the list of data sets.
        :param index: The index of the data set to retrieve.
        :return: The data set.
        """
        return self.datasets[index]

    def get_dataset_count(self) -> int:
        """Gets the number of data sets in the list.
        :return: The number of data sets.
        """
        return len(self.datasets)

    def remove_dataset(self, index):
        """Removes a data set from the list of data sets.
        :param index: The index of the data set to remove.
        """
        self.datasets.pop(index)

    def clear_datasets(self):
        """Clears the list of data sets."""
        self.datasets.clear()
