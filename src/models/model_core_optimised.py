# © 2024 Seb Garrioch. All rights reserved.
# Published under the MIT License.
import src.common as co
import cupy as cp
import pandas as pd

from src.data import DataSetManager, DatasetMetadata
from src.errors import ModelError
from typing import Optional

class ModelCoreOptimised:
    """Provides the core functionality for the SPNN Model using performance optimised code."""

    __column_wise_mean: cp.ndarray
    __column_wise_standard_deviation: cp.ndarray
    __dataset_handle: str
    __dataset_manager: Optional[DataSetManager]
    __dataset_metadata: Optional[DatasetMetadata]
    __input_size: int
    __learning_rate: float
    __output_size: int
    __parameters: cp.ndarray
    __training_setup_completed: bool

    def __init__(self):
        """Initialises the ModelCoreOptimised class."""
        self.__column_wise_mean = cp.array([])
        self.__column_wise_standard_deviation = cp.array([])
        self.__dataset_handle = ""
        self.__dataset_manager = None
        self.__dataset_metadata = None
        self.__input_size = 0
        self.__learning_rate = 0.0
        self.__output_size = 0
        self.__parameters = cp.array([])
        self.__training_setup_completed = False

    def get_dataset(self) -> pd.DataFrame:
        """Returns the dataset used for training.
        :return: The dataset used for training.
        """
        return self.__dataset_manager.get_dataset(self.__dataset_handle)

    def get_input_size(self) -> int:
        """Returns the number of features in the dataset.
        :return: The number of features in the dataset.
        """
        return self.__input_size

    def get_output_size(self) -> int:
        """Returns the number of labels in the dataset.
        :return: The number of labels in the dataset.
        """
        return self.__output_size

    def get_parameters(self) -> dict[str, cp.ndarray]:
        """Returns the parameters (weights and biases) for the SPNN model.
        :return: The parameters (weights and biases) for the SPNN model.
        """
        named_parameters = dict[str, cp.ndarray]()
        for i in range(self.__parameters.shape[1] - 1):
            named_parameters[co.WEIGHT_PARAMETER_PREFIX + str(i)] = self.__parameters[0, i]
        named_parameters[co.BIAS_PARAMETER] = self.__parameters[0, self.__parameters.shape[1] - 1]
        return named_parameters

    def get_training_setup_completed(self) -> bool:
        """Returns whether the training setup for the SPNN model has been completed.
        :return: True if the training setup has been completed; otherwise, False.
        """
        return self.__training_setup_completed

    def get_transposed_normalised_label(self) -> cp.ndarray:
        """Returns the transposed normalised label values from the dataset metadata.
        :return: The transposed normalised label values.
        """
        return self.__dataset_metadata.get_transposed_normalised_label()

    def back_propagation(self, y_hat: cp.ndarray) -> cp.ndarray:
        """Performs back propagation for the SPNN model.
        :param y_hat: The predicted output values.
        :return: The gradients of the parameters (weights and biases) with respect to the loss function.
        """
        # Aids in debugging.
        if not self.__training_setup_completed:  # pragma: no cover
            raise ModelError(co.EXCEPTION_MESSAGE_TRAINING_SETUP_NOT_COMPLETED)

        # Retrieve the normalised features from the dataset metadata.
        normalised_features = self.__dataset_metadata.get_transposed_normalised_features()

        # The number of samples in the dataset.
        m = y_hat.shape[1]

        # Retrieve the actual output values from the dataset metadata.
        y = self.__dataset_metadata.get_transposed_normalised_label()

        # Compute the gradients of the parameters (weights and bias) with respect to the loss function.
        dz = y_hat - y

        return cp.dot(dz, normalised_features.T) / m

    def flush_training_setup(self):
        """Flushes the training setup for the SPNN model but preserves the parameters."""
        self.__dataset_handle = ""
        self.__dataset_manager = None
        self.__dataset_metadata = None
        self.__training_setup_completed = False

    def forward_propagation(self) -> cp.ndarray:
        """Performs forward propagation for the SPNN model.
        :return: The proposed pre-activation (Z) values. In linear regression, the Z values are also called the weighted
        sums or the linear combinations of the input features.
        """
        # Aids in debugging.
        if not self.__training_setup_completed:  # pragma: no cover
            raise ModelError(co.EXCEPTION_MESSAGE_TRAINING_SETUP_NOT_COMPLETED)

        # Retrieve the normalised features from the dataset metadata.
        normalised_features = self.__dataset_metadata.get_transposed_normalised_features()
        # Get the parameters (weights and bias) for the SPNN model.
        parameters = self.__parameters
        # Compute the pre-activation values for each sample.
        return cp.dot(parameters, normalised_features)

    def predict(self, inference_data: pd.DataFrame) -> cp.ndarray:
        """Predicts the output values based on the input values using the SPNN model.
        :param inference_data: The input values for prediction.
        """
        # Normalise the input values using the column-wise mean and standard deviation.
        data_count = len(inference_data)
        feature_count = self.__input_size
        variable_count = feature_count + 1
        # The column-wise mean and standard deviation include the label column, which is why the data must be sliced.
        normalised_inference_data = ((inference_data.to_numpy() - self.__column_wise_mean[0:feature_count])
                                     / self.__column_wise_standard_deviation[0:feature_count])
        transposed_normalised_inference_data = cp.ndarray((variable_count, data_count))
        for i in range(feature_count):
            transposed_normalised_inference_data[i] = cp.array(normalised_inference_data[:, i])
        # Add a row of ones for the bias term, which facilitates matrix multiplication.
        transposed_normalised_inference_data[variable_count - 1] = cp.ones((1, data_count))

        # Get the parameters (weights and bias) for the SPNN model.
        parameters = self.__parameters
        # Perform forward propagation to predict the output values.
        z = cp.dot(parameters, transposed_normalised_inference_data)

        # De-normalise the output values using the column-wise mean and standard deviation.
        return z * self.__column_wise_standard_deviation[feature_count] + self.__column_wise_mean[feature_count]

    def setup_linear_regression_training(
            self
            , feature_names: list[str]
            , label_name: str
            , learning_rate: float
            , handle: str
            , file_name: str
            , competition_dataset: bool = False
            , dataset: Optional[pd.DataFrame] = None):
        """Sets up the training for the SPNN model using linear regression.
        :param feature_names: The list of feature columns in the dataset.
        :param label_name: The name of the label column in the dataset.
        changes by less than the epsilon value.
        :param learning_rate: The learning rate for training.
        :param handle: The dataset's Kaggle handle.
        :param file_name: The file name of the dataset, including the extension.
        :param competition_dataset: Whether the dataset is a competition dataset.
        :param dataset: The dataset to use for training. If None, the dataset is acquired from Kaggle.
        """
        self.__feature_names = feature_names
        self.__label_name = label_name
        self.__learning_rate = learning_rate
        # Initialise the parameters' dictionary.
        self.__parameters = {}
        # Reset the training setup flag.
        self.__training_setup_completed = False

        # Acquire the dataset. Cannot be tested due to Kaggle call requirement.
        self.__dataset_manager = DataSetManager()
        self.__dataset_handle = handle
        if dataset is None: # pragma: no cover
            self.__dataset_manager.acquire_kaggle_dataset(handle, file_name, competition_dataset)
            dataset = self.__dataset_manager.get_dataset(handle)
        else:
            self.__dataset_manager.add_dataset(handle, dataset)

        # Compute the metadata for the dataset.
        self.__dataset_metadata = DatasetMetadata(dataset, feature_names, label_name, False)
        self.__column_wise_mean = self.__dataset_metadata.get_column_wise_mean()
        self.__column_wise_standard_deviation = self.__dataset_metadata.get_column_wise_standard_deviation()
        # Column normalisation is used to ensure consistent scaling across all features and labels.
        self.__dataset_metadata.compute_column_wise_normalisation()
        # Transposing the normalised column vectors is used to facilitate matrix multiplication (dot product).
        self.__dataset_metadata.transpose_normalised_column_vectors()

        # The input size is the number of features in the dataset.
        self.__input_size = self.__dataset_metadata.get_feature_count()
        # The output size is the number of labels in the dataset.
        self.__output_size = self.__dataset_metadata.get_label_count()

        # Initialise the parameters (weights and bias) for the SPNN model.
        # There are input_size weights and one bias.
        parameter_count = self.__input_size + 1
        # The parameters are stored in a row vector.
        self.__parameters = cp.ndarray((self.__output_size, parameter_count))
        # The weights are initialised for each feature.
        for i in range(parameter_count - 1):
            # The weights are randomly initialised using a normal distribution.
            self.__parameters[0, i] = cp.random.randn(1, 1) * learning_rate

        # The biases are initialised to zero.
        self.__parameters[0, parameter_count - 1] = cp.zeros((1, self.__output_size))

        # The training setup is now complete.
        self.__training_setup_completed = True

    def update_parameters(self, gradients: cp.ndarray):
        """Updates the parameters (weights and biases) for the SPNN model.
        :param gradients: The gradients of the parameters (weights and biases) with respect to the loss function.
        """
        # Aids in debugging.
        if not self.__training_setup_completed:  # pragma: no cover
            raise ModelError(co.EXCEPTION_MESSAGE_TRAINING_SETUP_NOT_COMPLETED)

        # Update the parameters using the gradients and the learning rate.
        self.__parameters = self.__parameters - self.__learning_rate * gradients
