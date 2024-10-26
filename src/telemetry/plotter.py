# © 2024 Seb Garrioch. All rights reserved.
# Published under the MIT License.
import src.common as co
import cupy as cp
import matplotlib.pyplot as plt
import os
import pandas as pd

from typing import Optional


class Plotter: # pragma: no cover
    """Plots training data and incremental learning data to facilitate analysis of the training process."""

    __output_directory: Optional[str]

    def __init__(self, output_directory: str = None):
        """Initializes a new instance of the Plotter class.
        :param output_directory: The directory to save plots to.
        """
        if output_directory and not os.path.exists(os.path.normpath(output_directory)):
            raise ValueError(co.EXCEPTION_MESSAGE_DIRECTORY_DOES_NOT_EXIST_FORMAT.format(output_directory))
        self.__output_directory = output_directory

    def get_output_directory(self) -> str:
        """Returns the output directory for saving plots."""
        return self.__output_directory

    def plot_2d(
            self
            , dataset: pd.DataFrame
            , x_data_name: str
            , y_data_name: str
            , y_hat: cp.ndarray
            , plot_title: str
            , x_axis_label: str
            , y_axis_label: str
            , save_file_name: str = None
            , save_file: bool = False):
        """Plots the data and the linear regression line.
        :param dataset: The dataset to plot.
        :param x_data_name: The name of the x data column.
        :param y_data_name: The name of the y data column.
        :param y_hat: The predicted y values.
        :param plot_title: The title of the plot.
        :param x_axis_label: The label of the x-axis.
        :param y_axis_label: The label of the y-axis.
        :param save_file_name: The name of the file used to save the plot.
        :param save_file: A flag indicating whether to save the plot to a file.
        """
        if dataset is None or dataset.empty:
            raise ValueError(co.EXCEPTION_MESSAGE_NONE_OR_EMPTY_VALUE_FORMAT.format("dataset"))
        if not x_data_name:
            raise ValueError(co.EXCEPTION_MESSAGE_NONE_OR_EMPTY_VALUE_FORMAT.format("x_data_name"))
        if not y_data_name:
            raise ValueError(co.EXCEPTION_MESSAGE_NONE_OR_EMPTY_VALUE_FORMAT.format("y_data_name"))
        if y_hat is None or y_hat.size == 0:
            raise ValueError(co.EXCEPTION_MESSAGE_NONE_OR_EMPTY_VALUE_FORMAT.format("y_hat"))
        if not plot_title:
            raise ValueError(co.EXCEPTION_MESSAGE_NONE_OR_EMPTY_VALUE_FORMAT.format("plot_title"))
        if not x_axis_label:
            raise ValueError(co.EXCEPTION_MESSAGE_NONE_OR_EMPTY_VALUE_FORMAT.format("x_axis_label"))
        if not y_axis_label:
            raise ValueError(co.EXCEPTION_MESSAGE_NONE_OR_EMPTY_VALUE_FORMAT.format("y_axis_label"))
        if x_data_name not in dataset.columns:
            raise ValueError(co.EXCEPTION_MESSAGE_NOT_IN_DATA_FRAME_FORMAT.format(x_data_name))
        if y_data_name not in dataset.columns:
            raise ValueError(co.EXCEPTION_MESSAGE_NOT_IN_DATA_FRAME_FORMAT.format(y_data_name))

        # Add the title and axis labels.
        plt.title(plot_title)
        plt.xlabel(x_axis_label)
        plt.ylabel(y_axis_label)

        # Define the scatter plot of the data.
        plt.scatter(dataset[x_data_name], dataset[y_data_name], color='black')

        # Prepare the linear regression line data.
        y_hat_cpu = cp.asnumpy(y_hat).flatten()
        x_values = dataset[x_data_name].values
        sorted_indices = y_hat_cpu.argsort()
        y_hat_sorted = y_hat_cpu[sorted_indices]
        x_values_sorted = x_values[sorted_indices]

        # Plot
        plt.plot(x_values_sorted, y_hat_sorted, color="blue", label="Predicted Line (y_hat)")
        plt.legend()

        # Save the plot to a file if requested.
        if save_file and self.__output_directory and save_file_name:
            try:
                save_path = os.path.normpath(os.path.join(self.__output_directory, save_file_name))
                plt.savefig(save_path)
            except Exception as e:
                # Print an error message if the plot could not be saved, but do not raise an exception.
                print("\033[91mFailed to save plot to file: {0}\033[0m".format(e))
        else:
            plt.show()
        plt.close()
