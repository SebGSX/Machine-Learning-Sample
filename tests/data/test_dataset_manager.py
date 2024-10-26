# © 2024 Seb Garrioch. All rights reserved.
# Published under the MIT License.
import pandas as pd
import pytest

from src.data import DataSetManager

HANDLE: str = "test/testcsv"

@pytest.fixture
def manager() -> DataSetManager:
    """Fixture to create a DataSetManager instance.
    :returns: A DataSetManager instance.
    """
    return DataSetManager()

@pytest.fixture
def sample_data() -> pd.DataFrame:
    """Returns a sample dataset for testing."""
    data = {
        "Newspaper": [69.2, 45.1, 69.3],
        "Sales": [22.1, 10.4, 9.3]
    }
    return pd.DataFrame(data)

def test_get_datasets(manager: DataSetManager, sample_data: pd.DataFrame):
    """Test retrieving all datasets.
    :param manager: The DataSetManager instance.
    :param sample_data: The sample data for testing.
    """
    test_handle: str = "test/testcsv2"
    manager.add_dataset(HANDLE, sample_data)
    manager.add_dataset(test_handle, sample_data)
    assert manager.get_datasets() == {
                HANDLE: sample_data,
                test_handle: sample_data
           }

def test_get_dataset(manager: DataSetManager, sample_data: pd.DataFrame):
    """Test retrieving a specific dataset by index.
    :param manager: The DataSetManager instance.
    :param sample_data: The sample data for testing.
    """
    manager.add_dataset(HANDLE, sample_data)
    assert manager.get_dataset(HANDLE)["Newspaper"].tolist() == [69.2, 45.1, 69.3]
    assert manager.get_dataset(HANDLE)["Sales"].tolist() == [22.1, 10.4, 9.3]

def test_get_dataset_with_nonexistent(manager: DataSetManager):
    """Test retrieving a dataset that does not exist raises a KeyError.
    :param manager: The DataSetManager instance.
    """
    with pytest.raises(KeyError):
        manager.get_dataset(HANDLE)

def test_get_dataset_count(manager: DataSetManager, sample_data: pd.DataFrame):
    """Test counting the number of datasets.
    :param manager: The DataSetManager instance.
    :param sample_data: The sample data for testing.
    """
    test_handle: str = "test/testcsv2"
    manager.add_dataset(HANDLE, sample_data)
    manager.add_dataset(test_handle, sample_data)
    assert manager.get_dataset_count() == 2

def test_add_dataset(manager: DataSetManager, sample_data: pd.DataFrame):
    """Test adding a dataset.
    :param manager: The DataSetManager instance.
    :param sample_data: The sample data for testing.
    """
    manager.add_dataset(HANDLE, sample_data)
    assert manager.get_dataset(HANDLE)["Newspaper"].tolist() == [69.2, 45.1, 69.3]
    assert manager.get_dataset(HANDLE)["Sales"].tolist() == [22.1, 10.4, 9.3]

def test_add_dataset_with_duplicate(manager: DataSetManager, sample_data: pd.DataFrame):
    """Test adding a dataset.
    :param manager: The DataSetManager instance.
    :param sample_data: The sample data for testing.
    """
    manager.add_dataset(HANDLE, sample_data)
    try:
        manager.add_dataset(HANDLE, sample_data)
    except:
        assert False

def test_clear_datasets(manager: DataSetManager, sample_data: pd.DataFrame):
    """Test clearing all datasets.
    :param manager: The DataSetManager instance.
    :param sample_data: The sample data for testing.
    """
    test_handle: str = "test/testcsv2"
    manager.add_dataset(HANDLE, sample_data)
    manager.add_dataset(test_handle, sample_data)
    manager.clear_datasets()
    assert manager.get_datasets() == {}

def test_remove_dataset(manager: DataSetManager, sample_data: pd.DataFrame):
    """Test removing a dataset by index.
    :param manager: The DataSetManager instance.
    :param sample_data: The sample data for testing.
    """
    test_handle: str = "test/testcsv2"
    manager.add_dataset(HANDLE, sample_data)
    manager.add_dataset(test_handle, sample_data)
    manager.remove_dataset(test_handle)
    assert manager.get_datasets() == { HANDLE: sample_data }

def test_remove_dataset_with_nonexistent(manager: DataSetManager):
    """Test removing a dataset that does not exist.
    :param manager: The DataSetManager instance.
    """
    try:
        manager.remove_dataset(HANDLE)
    except:
        assert False
