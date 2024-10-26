# © 2024 Seb Garrioch. All rights reserved.
# Published under the MIT License.
import cupy as cp
import os
import pandas as pd
import pytest

from collections import namedtuple
from pytest_mock import MockerFixture
from src.telemetry import Plotter


SampleData = namedtuple(
    "SampleData"
    , ["data_frame", "feature_name", "label_name", "y_hat"])

@pytest.fixture
def sample_data():
    """Fixture providing a sample dataset for testing."""
    data = {
        "TV": [230.1, 44.5, 17.2],
        "Sales": [22.1, 10.4, 9.3]
    }
    return SampleData(pd.DataFrame(data), "TV", "Sales", cp.array([20.0, 10.0, 15.0]))

def test_initialisation():
    """Tests the initialisation of the Plotter class."""
    plotter = Plotter(os.getcwd())
    assert plotter.get_output_directory() == os.getcwd()

def test_initialisation_with_empty_output_directory():
    """Tests initialising the Plotter class with an empty output directory."""
    with pytest.raises(ValueError):
        Plotter("")

def test_initialisation_with_invalid_output_directory():
    """Tests initialising the Plotter class with an invalid output directory."""
    with pytest.raises(ValueError):
        Plotter("invalid")

def test_plot_2d_shows_plot(mocker: MockerFixture, sample_data: tuple[pd.DataFrame, str, str, cp.ndarray]):
    """Tests the plot_2d method of the Plotter class.
    :param mocker: The mocker fixture.
    :param sample_data: The sample data for testing.
    """
    dataset, feature_name, label_name, y_hat = sample_data
    mocked_show = mocker.patch("matplotlib.pyplot.show")
    mocked_close = mocker.patch("matplotlib.pyplot.close")
    plotter = Plotter(os.getcwd())
    plotter.plot_2d(
        dataset
        , feature_name
        , label_name
        , y_hat
        , "Test Plot"
        , "X"
        , "Y"
        , "test_plot.png"
        , False
    )
    mocked_show.assert_called_once()
    mocked_close.assert_called_once()

def test_plot_2d_saves_plot(mocker: MockerFixture, sample_data: tuple[pd.DataFrame, str, str, cp.ndarray]):
    """Tests the plot_2d method of the Plotter class.
    :param mocker: The mocker fixture.
    :param sample_data: The sample data for testing.
    """
    dataset, feature_name, label_name, y_hat = sample_data
    mocked_savefig = mocker.patch("matplotlib.pyplot.savefig")
    mocked_show = mocker.patch("matplotlib.pyplot.show")
    mocked_close = mocker.patch("matplotlib.pyplot.close")
    cwd = os.getcwd()
    save_file_name = "test_plot.png"
    save_path = os.path.normpath(os.path.join(cwd, save_file_name))
    plotter = Plotter(cwd)
    plotter.plot_2d(
        dataset
        , feature_name
        , label_name
        , y_hat
        , "Test Plot"
        , "X"
        , "Y"
        , save_file_name
        , True
    )
    mocked_savefig.assert_called_once_with(save_path)
    mocked_show.assert_not_called()
    mocked_close.assert_called_once()

def test_plot_2d_with_empty_dataset():
    """Tests that the plot_2d method of the Plotter class raises an exception with an empty dataset."""
    plotter = Plotter(os.getcwd())
    with pytest.raises(ValueError):
        plotter.plot_2d(
            pd.DataFrame()
            , "TV"
            , "Sales"
            , cp.array([])
            , "Test Plot"
            , "X"
            , "Y"
            , "test_plot.png"
            , False
        )

def test_plot_2d_with_empty_x_data_name():
    """Tests that the plot_2d method of the Plotter class raises an exception with an empty x data name."""
    plotter = Plotter(os.getcwd())
    with pytest.raises(ValueError):
        plotter.plot_2d(
            pd.DataFrame({"TV": [230.1, 44.5, 17.2]})
            , ""
            , "Sales"
            , cp.array([20.0, 10.0, 15.0])
            , "Test Plot"
            , "X"
            , "Y"
            , "test_plot.png"
            , False
        )

def test_plot_2d_with_empty_y_data_name():
    """Tests that the plot_2d method of the Plotter class raises an exception with an empty y data name."""
    plotter = Plotter(os.getcwd())
    with pytest.raises(ValueError):
        plotter.plot_2d(
            pd.DataFrame({"TV": [230.1, 44.5, 17.2]})
            , "TV"
            , ""
            , cp.array([20.0, 10.0, 15.0])
            , "Test Plot"
            , "X"
            , "Y"
            , "test_plot.png"
            , False
        )

def test_plot_2d_with_empty_y_hat():
    """Tests that the plot_2d method of the Plotter class raises an exception with empty y_hat values."""
    plotter = Plotter(os.getcwd())
    with pytest.raises(ValueError):
        plotter.plot_2d(
            pd.DataFrame({"TV": [230.1, 44.5, 17.2]})
            , "TV"
            , "Sales"
            , cp.array([])
            , "Test Plot"
            , "X"
            , "Y"
            , "test_plot.png"
            , False
        )

def test_plot_2d_with_empty_plot_title():
    """Tests that the plot_2d method of the Plotter class raises an exception with an empty plot title."""
    plotter = Plotter(os.getcwd())
    with pytest.raises(ValueError):
        plotter.plot_2d(
            pd.DataFrame({"TV": [230.1, 44.5, 17.2]})
            , "TV"
            , "Sales"
            , cp.array([20.0, 10.0, 15.0])
            , ""
            , "X"
            , "Y"
            , "test_plot.png"
            , False
        )

def test_plot_2d_with_empty_x_axis_label():
    """Tests that the plot_2d method of the Plotter class raises an exception with an empty x axis label."""
    plotter = Plotter(os.getcwd())
    with pytest.raises(ValueError):
        plotter.plot_2d(
            pd.DataFrame({"TV": [230.1, 44.5, 17.2]})
            , "TV"
            , "Sales"
            , cp.array([20.0, 10.0, 15.0])
            , "Test Plot"
            , ""
            , "Y"
            , "test_plot.png"
            , False
        )

def test_plot_2d_with_empty_y_axis_label():
    """Tests that the plot_2d method of the Plotter class raises an exception with an empty y axis label."""
    plotter = Plotter(os.getcwd())
    with pytest.raises(ValueError):
        plotter.plot_2d(
            pd.DataFrame({"TV": [230.1, 44.5, 17.2]})
            , "TV"
            , "Sales"
            , cp.array([20.0, 10.0, 15.0])
            , "Test Plot"
            , "X"
            , ""
            , "test_plot.png"
            , False
        )

def test_plot_2d_with_invalid_x_data_name():
    """Tests that the plot_2d method of the Plotter class raises an exception with an invalid x data name."""
    plotter = Plotter(os.getcwd())
    with pytest.raises(ValueError):
        plotter.plot_2d(
            pd.DataFrame({"TV": [230.1, 44.5, 17.2]})
            , "Invalid"
            , "Sales"
            , cp.array([20.0, 10.0, 15.0])
            , "Test Plot"
            , "X"
            , "Y"
            , "test_plot.png"
            , False
        )

def test_plot_2d_with_invalid_y_data_name():
    """Tests that the plot_2d method of the Plotter class raises an exception with an invalid y data name."""
    plotter = Plotter(os.getcwd())
    with pytest.raises(ValueError):
        plotter.plot_2d(
            pd.DataFrame({"TV": [230.1, 44.5, 17.2]})
            , "TV"
            , "Invalid"
            , cp.array([20.0, 10.0, 15.0])
            , "Test Plot"
            , "X"
            , "Y"
            , "test_plot.png"
            , False
        )

