# © 2024 Seb Garrioch. All rights reserved.
# Published under the MIT License.
import cupy as cp
import os as os
import pandas as pd
import pytest

from pytest_mock import MockerFixture
from src.models import SpnnModel, ModelCoreEducational


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

def test_initialisation():
    """Tests the initialisation of the SpnnModel class."""
    model = SpnnModel(ModelCoreEducational())
    assert not model.get_converged()
    assert not model.get_training_setup_completed()

def test_initialisation_with_valid_output_directory():
    """Tests the initialisation of the SpnnModel class with a valid output directory."""
    cwd = os.getcwd()
    model = SpnnModel(ModelCoreEducational(), cwd)
    assert not model.get_converged()
    assert not model.get_training_setup_completed()

def test_initialisation_with_nonexistent_output_directory(mocker: MockerFixture):
    """Tests the initialisation of the SpnnModel class with a nonexistent output directory."""
    mocked_makedirs = mocker.patch("os.makedirs")
    # Ensure that the Plotter class is not initialised to avoid raising an exception.
    mocker.patch("src.telemetry.plotter.Plotter.__init__").return_value = None
    output_directory = "nonexistent"
    model = SpnnModel(ModelCoreEducational(), output_directory)
    assert not model.get_converged()
    assert not model.get_training_setup_completed()
    mocked_makedirs.assert_called_once_with(output_directory)

def test_predict(sample_data: tuple[pd.DataFrame, list[str], str]):
    """Tests the predict method.
    :param sample_data: The sample data for testing.
    """
    df, feature_names, label_name = sample_data
    model = SpnnModel(ModelCoreEducational())
    model.setup_linear_regression_training(
        feature_names
        , label_name
        , 1e-8
        , 5
        , 100
        , 0.1
        , "devzohaib/tvmarketingcsv"
        , "tvmarketing.csv"
        , False
        , df)
    model.train_linear_regression()
    y_hat = model.predict(pd.DataFrame({ "TV": [4, 5, 6] }))

    actual1 = y_hat[0][0]
    actual2 = y_hat[0][1]
    actual3 = y_hat[0][2]

    assert cp.isclose(actual1, 4.0, rtol=1e-3)
    assert cp.isclose(actual2, 5.0, rtol=1e-3)
    assert cp.isclose(actual3, 6.0, rtol=1e-3)

def test_predict_with_training_setup_completed_not_converged(sample_data: tuple[pd.DataFrame, list[str], str]):
    """Tests the predict method when the model has not converged.
    :param sample_data: The sample data for testing.
    """
    df, feature_names, label_name = sample_data
    model = SpnnModel(ModelCoreEducational())
    model.setup_linear_regression_training(
        feature_names
        , label_name
        , 1e-8
        , 5
        , 4
        , 0.75
        , "devzohaib/tvmarketingcsv"
        , "tvmarketing.csv"
        , False
        , df)
    expected_message = "The model has not been trained."
    with pytest.raises(Exception) as exception_info:
        model.predict(pd.DataFrame({ "TV": [4, 5, 6] }))
    assert str(exception_info.value) == expected_message

def test_predict_with_training_setup_not_completed_not_converged(sample_data: tuple[pd.DataFrame, list[str], str]):
    """Tests the predict method when the model has not converged.
    :param sample_data: The sample data for testing.
    """
    model = SpnnModel(ModelCoreEducational())
    expected_message = "The model has not been trained."
    with pytest.raises(Exception) as exception_info:
        model.predict(pd.DataFrame({ "TV": [4, 5, 6] }))
    assert str(exception_info.value) == expected_message

def test_setup_linear_regression_training(sample_data: tuple[pd.DataFrame, list[str], str]):
    """Tests the setup_linear_regression_training method.
    :param sample_data: The sample data for testing.
    """
    df, feature_names, label_name = sample_data
    model = SpnnModel(ModelCoreEducational())
    model.setup_linear_regression_training(
        feature_names
        , label_name
        , 1e-8
        , 5
        , 100
        , 0.75
        , "devzohaib/tvmarketingcsv"
        , "tvmarketing.csv"
        , False
        , df)

    assert model.get_input_size() == len(feature_names)
    assert model.get_output_size() == 1
    assert model.get_training_setup_completed()

def test_train_linear_regression_with_negative_gradient(sample_data: tuple[pd.DataFrame, list[str], str]):
    """Tests the train_linear_regression method.
    :param sample_data: The sample data for testing.
    """
    df, feature_names, label_name = sample_data
    df = pd.DataFrame({ "TV": [-1, -2, -3], "Sales": [1, 2, 3] })
    model = SpnnModel(ModelCoreEducational())
    model.setup_linear_regression_training(
        feature_names
        , label_name
        , 1e-8
        , 5
        , 100
        , 0.1
        , "devzohaib/tvmarketingcsv"
        , "tvmarketing.csv"
        , False
        , df)
    model.train_linear_regression()

    assert model.get_converged()
    assert not model.get_training_setup_completed()
    assert cp.isclose(model.get_parameters()["W_0"], -1.0, rtol=1e-3)
    assert cp.isclose(model.get_parameters()["b"], 0.0, rtol=1e-3)

def test_train_linear_regression_with_positive_gradient(sample_data: tuple[pd.DataFrame, list[str], str]):
    """Tests the train_linear_regression method.
    :param sample_data: The sample data for testing.
    """
    df, feature_names, label_name = sample_data
    model = SpnnModel(ModelCoreEducational())
    model.setup_linear_regression_training(
        feature_names
        , label_name
        , 1e-8
        , 5
        , 100
        , 0.1
        , "devzohaib/tvmarketingcsv"
        , "tvmarketing.csv"
        , False
        , df)
    model.train_linear_regression()

    assert model.get_converged()
    assert not model.get_training_setup_completed()
    assert cp.isclose(model.get_parameters()["W_0"], 1.0, rtol=1e-3)
    assert cp.isclose(model.get_parameters()["b"], 0.0, rtol=1e-3)

def test_train_linear_regression_with_no_convergence(sample_data: tuple[pd.DataFrame, list[str], str]):
    """Tests the train_linear_regression method.
    :param sample_data: The sample data for testing.
    """
    df, feature_names, label_name = sample_data
    model = SpnnModel(ModelCoreEducational())
    model.setup_linear_regression_training(
        feature_names
        , label_name
        , 1e-8
        , 5
        , 4
        , 0.1
        , "devzohaib/tvmarketingcsv"
        , "tvmarketing.csv"
        , False
        , df)
    model.train_linear_regression()

    assert not model.get_converged()

def test_train_linear_regression_with_no_training_setup(sample_data: tuple[pd.DataFrame, list[str], str]):
    """Tests the train_linear_regression method.
    :param sample_data: The sample data for testing.
    """
    model = SpnnModel(ModelCoreEducational())
    expected_message = "The training setup has not been completed."
    with pytest.raises(Exception) as exception_info:
        model.train_linear_regression()
    assert str(exception_info.value) == expected_message

def test_train_linear_regression_with_plot_results(
        mocker: MockerFixture
        , sample_data: tuple[pd.DataFrame, list[str], str]):
    """Tests the train_linear_regression method with plotting enabled.
    :param mocker: The mocker fixture for mocking the Plotter class.
    :param sample_data: The sample data for testing.
    """
    df, feature_names, label_name = sample_data
    mocked_plot = mocker.patch("src.telemetry.plotter.Plotter.plot_2d")
    cwd = os.getcwd()
    model = SpnnModel(ModelCoreEducational(),cwd)
    model.setup_linear_regression_training(
        feature_names
        , label_name
        , 1e-8
        , 5
        , 100
        , 0.1
        , "devzohaib/tvmarketingcsv"
        , "tvmarketing.csv"
        , False
        , df)
    model.train_linear_regression(plot_results=True)
    assert mocked_plot.call_count >= 2

def test_train_linear_regression_with_plot_results_and_no_output_directory(mocker: MockerFixture
        , sample_data: tuple[pd.DataFrame, list[str], str]):
    """Tests the train_linear_regression method with plotting enabled and no output directory.
    :param mocker: The mocker fixture for mocking the Plotter class.
    :param sample_data: The sample data for testing.
    """
    df, feature_names, label_name = sample_data
    mocked_plot = mocker.patch("src.telemetry.plotter.Plotter.plot_2d")
    model = SpnnModel(ModelCoreEducational())
    model.setup_linear_regression_training(
        feature_names
        , label_name
        , 1e-8
        , 5
        , 100
        , 0.1
        , "devzohaib/tvmarketingcsv"
        , "tvmarketing.csv"
        , False
        , df)
    model.train_linear_regression(plot_results=True)
    assert mocked_plot.call_count == 1
