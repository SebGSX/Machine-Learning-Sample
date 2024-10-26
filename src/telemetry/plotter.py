# © 2024 Seb Garrioch. All rights reserved.
# Published under the MIT License.
import cupy as cp
import matplotlib.pyplot as plt
import pandas as pd


class Plotter: # pragma: no cover
    """Plots training data and incremental learning data to facilitate analysis of the training process."""

    __output_directory: str

    def __init__(self, output_directory: str = None):
        self.__output_directory = output_directory

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
        plt.plot(x_values_sorted, y_hat_sorted, color='blue', label='Predicted Line (y_hat)')
        plt.legend()

        # Save the plot to a file if requested.
        if save_file and self.__output_directory is not None and save_file_name is not None:
            try:
                plt.savefig(self.__output_directory + save_file_name)
            except Exception as e:
                print('Failed to save plot to file: ', str(e))
        else:
            plt.show()
