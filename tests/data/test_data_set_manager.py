# © 2024 Seb Garrioch. All rights reserved.
# Published under the MIT License.
import pytest
from data.dataset_manager import DataSetManager

@pytest.fixture
def manager():
    """Fixture to create a DataSetManager instance."""
    return DataSetManager()

def test_add_dataset(manager):
    """Test adding a data set."""
    manager.add_dataset("dataset_1")
    assert manager.get_datasets() == ["dataset_1"]

def test_get_datasets(manager):
    """Test retrieving all data sets."""
    manager.add_dataset("dataset_1")
    manager.add_dataset("dataset_2")
    assert manager.get_datasets() == ["dataset_1", "dataset_2"]

def test_get_dataset(manager):
    """Test retrieving a specific data set by index."""
    manager.add_dataset("dataset_1")
    assert manager.get_dataset(0) == "dataset_1"

def test_get_dataset_count(manager):
    """Test counting the number of data sets."""
    manager.add_dataset("dataset_1")
    manager.add_dataset("dataset_2")
    assert manager.get_dataset_count() == 2

def test_remove_dataset(manager):
    """Test removing a data set by index."""
    manager.add_dataset("dataset_1")
    manager.add_dataset("dataset_2")
    manager.remove_dataset(0)
    assert manager.get_datasets() == ["dataset_2"]

def test_clear_datasets(manager):
    """Test clearing all data sets."""
    manager.add_dataset("dataset_1")
    manager.add_dataset("dataset_2")
    manager.clear_datasets()
    assert manager.get_datasets() == []
