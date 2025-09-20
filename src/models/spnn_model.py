# © 2025 Seb Garrioch. All rights reserved.
# Published under the MIT License.
import src.common as co
import cupy as cp
import os
import pandas as pd

from datetime import datetime
from src.models import ModelCoreEducational, ModelCoreOptimised
from src.errors import ModelError
from typing import Optional, Union
from src.telemetry import Plotter


class SpnnModel:
    """A single perceptron neural network (SPNN) "spin" model used for educational purposes. The SPNN model uses linear
    regression and supervised learning to predict the output (label) based on the input (features) in a dataset.
    """

    __MODEL_OUTPUT_DIRECTORY_NOT_CREATED_FORMAT = "\033[91mFailed to create output directory: {0}\033[0m"
    __PLOT_FILE_NAME_FORMAT = "epoch-{0}.png"
    __PLOT_TITLE = "Linear Regression"

    __converged: bool
    __convergence_epsilon: float
    __convergence_patience: int
    __epochs: int
    __feature_names: list[str]
    __label_name: str
    __model_core: Union[ModelCoreEducational, ModelCoreOptimised]
    __plotter: Plotter

    def __init__(self, model_core: Union[ModelCoreEducational, ModelCoreOptimised], output_directory: str = None):
        """Initialises the SPNN model.
        :param model_core: The core of the model used by the SPNN model.
        :param output_directory: The directory for storing the output files.
        """
        try:
            if output_directory is not None and output_directory != "" and not os.path.exists(output_directory):
                os.makedirs(output_directory)
            self.__output_directory = os.path.abspath(output_directory)
            self.__output_directory_available = True
        except Exception as e:
            print(self.__MODEL_OUTPUT_DIRECTORY_NOT_CREATED_FORMAT.format(e))
            self.__output_directory = None
            self.__output_directory_available = False

        self.__converged = False
        self.__convergence_epsilon = 0.0
        self.__convergence_patience = 0
        self.__epochs = 0
        self.__feature_names = []
        self.__label_name = ""
        self.__model_core = model_core
        self.__plotter = Plotter(self.__output_directory)

    def __compute_sum_of_squares_cost(self, y_hat: cp.ndarray) -> float:
        """Computes the sum of squares cost for the SPNN model.
        :param y_hat: The predicted output values.
        :return: The sum of squares cost scaled by twice the number of samples.
        """
        # Aids in debugging.
        if not self.__model_core.get_training_setup_completed(): # pragma: no cover
            raise ModelError(co.EXCEPTION_MESSAGE_TRAINING_SETUP_NOT_COMPLETED)

        # Retrieve the actual output values from the dataset metadata.
        y = self.__model_core.get_transposed_normalised_label()

        # The cost is computed as the sum of the squared differences between the predicted and actual output values,
        # divided by twice the number of samples. The division by twice the number of samples is used to facilitate
        # the gradient descent algorithm.
        return cp.sum((y_hat - y) ** 2) / (2 * y.shape[1])

    def __flush_training_setup(self):
        """Flushes the training setup for the SPNN model."""
        self.__model_core.flush_training_setup()
        self.__convergence_epsilon = 0.0
        self.__convergence_patience = 0
        self.__epochs = 0

    def get_converged(self) -> bool:
        """Returns whether the SPNN model has converged.
        :return: True if the model has converged; otherwise, False.
        """
        return self.__converged

    def get_input_size(self) -> int:
        """Returns the number of features in the dataset.
        :return: The number of features in the dataset.
        """
        return self.__model_core.get_input_size()

    def get_output_size(self) -> int:
        """Returns the number of labels in the dataset.
        :return: The number of labels in the dataset.
        """
        return self.__model_core.get_output_size()

    def get_parameters(self) -> dict[str, cp.ndarray]:
        """Returns the parameters (weights and biases) for the SPNN model.
        :return: The parameters (weights and biases) for the SPNN model.
        """
        return self.__model_core.get_parameters()

    def get_training_setup_completed(self) -> bool:
        """Returns whether the training setup for the SPNN model has been completed.
        :return: True if the training setup has been completed; otherwise, False.
        """
        return self.__model_core.get_training_setup_completed()

    def predict(self, inference_data: pd.DataFrame) -> cp.ndarray:
        """Predicts the output values based on the input values using the SPNN model.
        :param inference_data: The input values for prediction.
        """
        if not self.__converged:
            raise ModelError(co.EXCEPTION_MESSAGE_MODEL_NOT_TRAINED)

        return self.__model_core.predict(inference_data)

    def setup_linear_regression_training(
            self
            , feature_names: list[str]
            , label_name: str
            , convergence_epsilon: float
            , convergence_patience: int
            , epochs: int
            , learning_rate: float
            , handle: str
            , file_name: str
            , competition_dataset: bool = False
            , dataset: Optional[pd.DataFrame] = None):
        """Sets up the training for the SPNN model using linear regression.
        :param feature_names: The list of feature columns in the dataset.
        :param label_name: The name of the label column in the dataset.
        :param convergence_epsilon: The epsilon value for the cost function to determine convergence.
        :param convergence_patience: The number of epochs to wait before determining convergence when the cost function
        changes by less than the epsilon value.
        :param epochs: The number of epochs for training.
        :param learning_rate: The learning rate for training.
        :param handle: The dataset's Kaggle handle.
        :param file_name: The file name of the dataset, including the extension.
        :param competition_dataset: Whether the dataset is a competition dataset.
        :param dataset: The dataset to use for training. If None, the dataset is acquired from Kaggle.
        """
        self.__convergence_epsilon = convergence_epsilon
        self.__convergence_patience = convergence_patience
        self.__epochs = epochs
        self.__feature_names = feature_names
        self.__label_name = label_name
        self.__model_core.setup_linear_regression_training(
            feature_names
            , label_name
            , learning_rate
            , handle
            , file_name
            , competition_dataset
            , dataset)

    def train_linear_regression(
            self
            , plot_results: bool = False):
        """Trains the SPNN model using linear regression.
        :param plot_results: Whether to plot the results of the training process.
        """
        if not self.__model_core.get_training_setup_completed():
            raise ModelError(co.EXCEPTION_MESSAGE_TRAINING_SETUP_NOT_COMPLETED)

        training_start_time = datetime.now()
        converged = False
        last_cost = 0.0
        last_iteration = 0
        same_last_cost_count = 0
        for i in range(self.__epochs):
            y_hat = self.__model_core.forward_propagation()
            cost = self.__compute_sum_of_squares_cost(y_hat)

            # Check for convergence.
            if cp.isclose(cost, last_cost, atol=self.__convergence_epsilon):
                same_last_cost_count += 1
                if same_last_cost_count == self.__convergence_patience:
                    converged = True
                    last_cost = cost
                    last_iteration = i
                    break
            else:
                last_cost = cost
                last_iteration = i
                same_last_cost_count = 0

            gradients = self.__model_core.back_propagation(y_hat)
            self.__model_core.update_parameters(gradients)

            # Plot the results of the training process, if requested.
            if plot_results and self.__model_core.get_input_size() == 1 and self.__output_directory_available:
                dataset = self.__model_core.get_dataset()
                feature_name = self.__feature_names[0]
                self.__plotter.plot_2d(
                    dataset
                    , feature_name
                    , self.__label_name
                    , self.__model_core.predict(dataset)
                    , self.__PLOT_TITLE
                    , feature_name
                    , self.__label_name
                    , self.__PLOT_FILE_NAME_FORMAT.format(i)
                    , True)

        self.__converged = converged
        training_stop_time = datetime.now()

        # Plot the results of the training process.
        if plot_results and self.__model_core.get_input_size() == 1:
            dataset = self.__model_core.get_dataset()
            feature_name = self.__feature_names[0]
            self.__plotter.plot_2d(
                dataset
                , feature_name
                , self.__label_name
                , self.__model_core.predict(dataset)
                , self.__PLOT_TITLE
                , feature_name
                , self.__label_name)

        # Prepare the model for prediction. Once flushed, the model can be saved and reloaded.
        if converged:
            self.__flush_training_setup()

        # Print the training results.
        duration = training_stop_time - training_start_time
        print("\n\nTraining results:")
        print(f"\tTraining time:\t\t\t {duration} (hours:minutes:seconds.milliseconds)")
        print("\tConvergence achieved:\t", converged)
        print("\tFinal cost:\t\t\t\t", last_cost)
        print("\tRequired iterations:\t", last_iteration + 1)
        print("\tFinal weights:")
        parameters = self.__model_core.get_parameters()
        for i in range(self.__model_core.get_input_size()):
            print(f"\t\t{co.WEIGHT_PARAMETER_PREFIX + str(i)}:\t\t\t\t"
                  , parameters[co.WEIGHT_PARAMETER_PREFIX + str(i)].item())
        print("\tFinal bias:\t\t\t\t", parameters[co.BIAS_PARAMETER].item())
        print("")
