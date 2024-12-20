# © 2024 Seb Garrioch. All rights reserved.
# Published under the MIT License.
import src.common as co
import cupy as cp
import pandas as pd

from src.data import DataSetManager, DatasetMetadata
from src.errors import ModelError
from typing import Optional

class ModelCoreEducational:
    """Provides the core functionality for the SPNN Model in an accessible manner, for educational purposes."""

    __column_wise_mean: pd.DataFrame
    __column_wise_standard_deviation: pd.DataFrame
    __dataset_handle: str
    __dataset_manager: Optional[DataSetManager]
    __dataset_metadata: Optional[DatasetMetadata]
    __feature_names: list[str]
    __input_size: int
    __label_name: str
    __learning_rate: float
    __output_size: int
    __parameters: dict[str, cp.ndarray]
    __training_setup_completed: bool

    def __init__(self):
        """Initialises the ModelCoreEducational class."""
        self.__column_wise_mean = cp.array([])
        self.__column_wise_standard_deviation = cp.array([])
        self.__dataset_handle = ""
        self.__dataset_manager = None
        self.__dataset_metadata = None
        self.__feature_names = []
        self.__input_size = 0
        self.__label_name = ""
        self.__learning_rate = 0.0
        self.__output_size = 0
        self.__parameters = {}
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
        return self.__parameters

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

    def back_propagation(self, y_hat: cp.ndarray) -> dict[str, cp.ndarray]:
        """Performs back propagation for the SPNN model.
        :param y_hat: The predicted output values.
        :return: The gradients of the parameters (weights and biases) with respect to the loss function.
        """
        # Aids in debugging.
        if not self.__training_setup_completed: # pragma: no cover
            raise ModelError(co.EXCEPTION_MESSAGE_TRAINING_SETUP_NOT_COMPLETED)

        # The number of samples in the dataset.
        m = y_hat.shape[1]

        # Retrieve the actual output values from the dataset metadata.
        y = self.__dataset_metadata.get_transposed_normalised_label()

        # Compute the gradients of the parameters (weights and bias) with respect to the loss function.
        gradients = {}
        dz = y_hat - y
        feature_count = self.__dataset_metadata.get_feature_count()
        for i in range(feature_count):
            # Retrieve the feature name and the normalised feature values.
            feature_name = self.__feature_names[i]
            normalised_feature = self.__dataset_metadata.get_transposed_normalised_features()[feature_name]
            gradients[co.PARTIAL_DERIVATIVE_WEIGHT_PARAMETER_PREFIX + str(i)] = cp.dot(dz, normalised_feature.T) / m

        gradients[co.PARTIAL_DERIVATIVE_BIAS_PARAMETER] = cp.sum(dz, axis=1, keepdims=True) / m

        return gradients

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
        if not self.__training_setup_completed: # pragma: no cover
            raise ModelError(co.EXCEPTION_MESSAGE_TRAINING_SETUP_NOT_COMPLETED)

        # Retrieve the normalised features from the dataset metadata.
        normalised_features = self.__dataset_metadata.get_transposed_normalised_features()
        # The number of samples in the dataset.
        m = normalised_features[self.__feature_names[0]].shape[1]
        # Retrieve the bias parameter.
        param_b = self.__parameters[co.BIAS_PARAMETER]
        # Initialise the pre-activation values to zero.
        z: cp.ndarray = cp.zeros((1, m))
        feature_count = self.__dataset_metadata.get_feature_count()
        for i in range(feature_count):
            # Retrieve the feature name and the weight parameter for the feature.
            feature_name = self.__feature_names[i]
            param_w = self.__parameters[co.WEIGHT_PARAMETER_PREFIX + str(i)]
            # Compute the pre-activation (Z) values per feature, then sum them across all features.
            z = z + cp.matmul(param_w, normalised_features[feature_name])

        # Add the bias parameter to the pre-activation values to obtain the final Z values.
        return z + param_b

    def predict(self, inference_data: pd.DataFrame) -> cp.ndarray:
        """Predicts the output values based on the input values using the SPNN model.
        :param inference_data: The input values for prediction.
        """
        # Normalise the input values using the column-wise mean and standard deviation. This relies on the named columns
        # being consistent with the dataset used for training so that the appropriate column-wise mean and standard
        # deviation values can be retrieved and applied to the inference data.
        normalised_inference_data = (inference_data - self.__column_wise_mean) / self.__column_wise_standard_deviation
        transposed_normalised_inference_data: dict[str, cp.array] = {}
        # Transpose the normalised input values to facilitate matrix multiplication (dot product).
        for feature_name in self.__feature_names:
            norm = normalised_inference_data[feature_name]
            transposed_normalised_inference_data[feature_name] = cp.array(norm).reshape(1, len(norm))

        # The number of samples in the inference data.
        m = transposed_normalised_inference_data[self.__feature_names[0]].shape[1]

        # Perform forward propagation to predict the output values.
        z: cp.ndarray = cp.zeros((1, m))
        for i in range(self.__input_size):
            param_w = self.__parameters[co.WEIGHT_PARAMETER_PREFIX + str(i)]
            z = z + cp.matmul(param_w, transposed_normalised_inference_data[self.__feature_names[i]])

        z: cp.ndarray = z + self.__parameters[co.BIAS_PARAMETER]

        # De-normalise the output values using the column-wise mean and standard deviation.
        return z * self.__column_wise_standard_deviation[self.__label_name] + self.__column_wise_mean[self.__label_name]

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
        self.__dataset_metadata = DatasetMetadata(dataset, feature_names, label_name)
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
        # The weights are initialised for each feature.
        for i in range(len(feature_names)):
            # The weights are randomly initialised using a normal distribution.
            param_w = cp.random.randn(1, 1) * learning_rate
            self.__parameters[co.WEIGHT_PARAMETER_PREFIX + str(i)] = param_w

        # The biases are initialised to zero.
        self.__parameters[co.BIAS_PARAMETER] = cp.zeros((1, self.__output_size))

        # The training setup is now complete.
        self.__training_setup_completed = True

    def update_parameters(self, gradients: dict[str, cp.ndarray]):
        """Updates the parameters (weights and biases) for the SPNN model.
        :param gradients: The gradients of the parameters (weights and biases) with respect to the loss function.
        """
        # Aids in debugging.
        if not self.__training_setup_completed: # pragma: no cover
            raise ModelError(co.EXCEPTION_MESSAGE_TRAINING_SETUP_NOT_COMPLETED)

        # Update the bias parameter: b = b - learning_rate * db.
        self.__parameters[co.BIAS_PARAMETER] = \
            self.__parameters[co.BIAS_PARAMETER] \
            - self.__learning_rate * gradients[co.PARTIAL_DERIVATIVE_BIAS_PARAMETER]

        # Update the weight parameters: W_i = W_i - learning_rate * dW_i.
        feature_count = self.__dataset_metadata.get_feature_count()
        for i in range(feature_count):
            self.__parameters[co.WEIGHT_PARAMETER_PREFIX + str(i)] = \
                self.__parameters[co.WEIGHT_PARAMETER_PREFIX + str(i)] - \
                self.__learning_rate * gradients[co.PARTIAL_DERIVATIVE_WEIGHT_PARAMETER_PREFIX + str(i)]
