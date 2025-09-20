# © 2025 Seb Garrioch. All rights reserved.
# Published under the MIT License.
import cupy as cp
import pandas as pd
import pytest

from src.models import ModelCoreEducational, ModelCoreOptimised
from typing import Union


@pytest.fixture
def sample_data():
    """Returns a sample dataset for testing."""
    data = {
        "TV": [1, 2, 3],
        "Sales": [1, 2, 3]
    }
    df = pd.DataFrame(data)
    feature_names = ["TV"]
    label_name = "Sales"
    return df, feature_names, label_name

@pytest.mark.parametrize("model_core", [ModelCoreEducational(), ModelCoreOptimised()])
def test_get_parameters_returns_keyed_data(
        model_core: Union[ModelCoreEducational, ModelCoreOptimised]
        , sample_data: tuple[pd.DataFrame, list[str], str]):
    """Tests that the get_parameters method returns keyed data.
    :param model_core: The model core to test.
    :param sample_data: The sample data for testing.
    """
    df, feature_names, label_name = sample_data
    model_core.setup_linear_regression_training(
        feature_names
        , label_name
        , 1e-8
        , "devzohaib/tvmarketingcsv"
        , "tvmarketing.csv"
        , False
        , df)
    actual = model_core.get_parameters()
    assert len(actual) == 2
    assert actual["W_0"] is not None
    assert actual["b"] is not None

@pytest.mark.parametrize("model_core", [ModelCoreEducational(), ModelCoreOptimised()])
def test_forward_propagation(
        model_core: Union[ModelCoreEducational, ModelCoreOptimised]
        , sample_data: tuple[pd.DataFrame, list[str], str]):
    """Tests the forward_propagation method.
    :param model_core: The model core to test.
    :param sample_data: The sample data for testing.
    """
    df, feature_names, label_name = sample_data
    model_core.setup_linear_regression_training(
        feature_names
        , label_name
        , 1e-8
        , "devzohaib/tvmarketingcsv"
        , "tvmarketing.csv"
        , False
        , df)
    actual = model_core.forward_propagation()

    # No training has been performed, so the forward propagation will yield values based on the initial weights.
    assert len(actual[0]) == 3

@pytest.mark.parametrize("model_core", [ModelCoreEducational(), ModelCoreOptimised()])
def test_flush_training_setup(
        model_core: Union[ModelCoreEducational, ModelCoreOptimised]
        , sample_data: tuple[pd.DataFrame, list[str], str]):
    """Tests the flush_training_setup method.
    :param model_core: The model core to test.
    :param sample_data: The sample data for testing.
    """
    df, feature_names, label_name = sample_data
    model_core.setup_linear_regression_training(
        feature_names
        , label_name
        , 1e-8
        , "devzohaib/tvmarketingcsv"
        , "tvmarketing.csv"
        , False
        , df)
    assert model_core.get_training_setup_completed()
    model_core.flush_training_setup()
    assert not model_core.get_training_setup_completed()

@pytest.mark.parametrize("model_core", [ModelCoreEducational(), ModelCoreOptimised()])
def test_predict(
        model_core: Union[ModelCoreEducational, ModelCoreOptimised]
        , sample_data: tuple[pd.DataFrame, list[str], str]):
    """Tests the predict method.
    :param model_core: The model core to test.
    :param sample_data: The sample data for testing.
    """
    df, feature_names, label_name = sample_data
    model_core.setup_linear_regression_training(
        feature_names
        , label_name
        , 1e-8
        , "devzohaib/tvmarketingcsv"
        , "tvmarketing.csv"
        , False
        , df)
    y_hat = model_core.predict(pd.DataFrame({ "TV": [4, 5, 6] }))

    actual1 = y_hat[0][0]
    actual2 = y_hat[0][1]
    actual3 = y_hat[0][2]

    # No training has been performed, so the prediction yields approximately 2.0 for all cases.
    assert cp.isclose(actual1, 2.0, rtol=1e-3)
    assert cp.isclose(actual2, 2.0, rtol=1e-3)
    assert cp.isclose(actual3, 2.0, rtol=1e-3)

@pytest.mark.parametrize("model_core", [ModelCoreEducational(), ModelCoreOptimised()])
def test_setup_linear_regression_training(
        model_core: Union[ModelCoreEducational, ModelCoreOptimised]
        , sample_data: tuple[pd.DataFrame, list[str], str]):
    """Tests the setup_linear_regression_training method.
    :param model_core: The model core to test.
    :param sample_data: The sample data for testing.
    """
    df, feature_names, label_name = sample_data
    model_core.setup_linear_regression_training(
        feature_names
        , label_name
        , 1e-8
        , "devzohaib/tvmarketingcsv"
        , "tvmarketing.csv"
        , False
        , df)

    assert model_core.get_dataset() is not None
    assert model_core.get_input_size() == len(feature_names)
    assert model_core.get_output_size() == 1
    assert model_core.get_training_setup_completed()