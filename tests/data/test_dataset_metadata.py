# © 2024 Seb Garrioch. All rights reserved.
# Published under the MIT License.
import cupy as cp
import pandas as pd
import pytest

from src.data import DatasetMetadata

@pytest.fixture
def sample_data():
    """Returns a sample dataset for testing."""
    data = {
        "TV": [230.1, 44.5, 17.2],
        "Radio": [37.8, 39.3, 45.9],
        "Newspaper": [69.2, 45.1, 69.3],
        "Sales": [22.1, 10.4, 9.3]
    }
    df = pd.DataFrame(data)
    feature_names = ["TV", "Radio", "Newspaper"]
    label_names = ["Sales"]
    return df, feature_names, label_names

def test_initialization(sample_data: pd.DataFrame):
    """Tests the initialisation of the DatasetMetadata class.
    :param sample_data: The sample data for testing.
    """
    df, feature_names, label_names = sample_data
    metadata = DatasetMetadata(df, feature_names, label_names)
    assert metadata.get_column_wise_normalisation().empty
    assert metadata.get_feature_count() == 3
    assert metadata.get_label_count() == 1

def test_get_column_wise_normalisation(sample_data: pd.DataFrame):
    """Tests the get_column_wise_normalisation method.
    :param sample_data: The sample data for testing.
    """
    df, feature_names, label_names = sample_data
    metadata = DatasetMetadata(df, feature_names, label_names)
    metadata.compute_column_wise_normalisation()
    normalised_df = metadata.get_column_wise_normalisation()
    assert not normalised_df.empty

def test_get_feature_count(sample_data: pd.DataFrame):
    """Tests the get_feature_count method.
    :param sample_data: The sample data for testing.
    """
    df, feature_names, label_names = sample_data
    metadata = DatasetMetadata(df, feature_names, label_names)
    assert metadata.get_feature_count() == len(feature_names)

def test_get_label_count(sample_data: pd.DataFrame):
    """Tests the get_label_count method.
    :param sample_data: The sample data for testing.
    """
    df, feature_names, label_names = sample_data
    metadata = DatasetMetadata(df, feature_names, label_names)
    assert metadata.get_label_count() == len(label_names)

def test_compute_column_wise_normalisation(sample_data: pd.DataFrame):
    """Tests the computation of column-wise normalisation.
    :param sample_data: The sample data for testing.
    """
    df, feature_names, label_names = sample_data
    metadata = DatasetMetadata(df, feature_names, label_names)
    metadata.compute_column_wise_normalisation()
    normalised_df = metadata.get_column_wise_normalisation()
    assert not normalised_df.empty
    assert metadata.get_column_wise_normalisation_computed()

def test_compute_column_wise_normalisation_force_recompute(sample_data: pd.DataFrame):
    """Tests the computation of column-wise normalisation with force_recompute=True.
    :param sample_data: The sample data for testing.
    """
    df, feature_names, label_names = sample_data
    metadata = DatasetMetadata(df, feature_names, label_names)
    metadata.compute_column_wise_normalisation()
    metadata.compute_column_wise_normalisation(force_recompute=True)
    normalised_df = metadata.get_column_wise_normalisation()
    assert not normalised_df.empty
    assert metadata.get_column_wise_normalisation_computed()

def test_compute_column_wise_normalisation_without_recompute(sample_data: pd.DataFrame):
    """Tests the computation of column-wise normalisation without recompute.
    :param sample_data: The sample data for testing.
    """
    df, feature_names, label_names = sample_data
    metadata = DatasetMetadata(df, feature_names, label_names)
    metadata.compute_column_wise_normalisation()
    metadata.compute_column_wise_normalisation()
    normalised_df = metadata.get_column_wise_normalisation()
    assert not normalised_df.empty
    assert metadata.get_column_wise_normalisation_computed()

def test_transpose_normalised_column_vectors(sample_data: pd.DataFrame):
    """Tests the transposition of normalised column vectors.
    :param sample_data: The sample data for testing.
    """
    df, feature_names, label_names = sample_data
    metadata = DatasetMetadata(df, feature_names, label_names)
    metadata.compute_column_wise_normalisation()
    metadata.transpose_normalised_column_vectors()
    transposed_features = metadata.get_transposed_normalised_features()
    transposed_label = metadata.get_transposed_normalised_label()
    assert all(isinstance(v, cp.ndarray) for v in transposed_features.values())
    assert isinstance(transposed_label, cp.ndarray)

def test_transpose_normalised_column_vectors_without_prior_normalisation(sample_data: pd.DataFrame):
    """Tests the transposition of normalised column vectors without prior normalisation.
    :param sample_data: The sample data for testing.
    """
    df, feature_names, label_names = sample_data
    metadata = DatasetMetadata(df, feature_names, label_names)
    flag_before = metadata.get_column_wise_normalisation_computed()
    metadata.transpose_normalised_column_vectors()
    transposed_features = metadata.get_transposed_normalised_features()
    transposed_label = metadata.get_transposed_normalised_label()
    assert all(isinstance(v, cp.ndarray) for v in transposed_features.values())
    assert isinstance(transposed_label, cp.ndarray)
    assert not flag_before
    assert metadata.get_column_wise_normalisation_computed()
