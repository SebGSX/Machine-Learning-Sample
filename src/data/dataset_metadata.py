# © 2024 Seb Garrioch. All rights reserved.
# Published under the MIT License.
from typing import Union

import src.common as co
import cupy as cp
import pandas as pd


class DatasetMetadata:
    """Represents metadata about a dataset in tabular format. Columns are expected to represent features or labels while
    rows are expected to represent individual events, experiments, instances, observations, samples, etc."""

    __active_columns_names: list[str]
    __column_wise_mean_keyed: pd.DataFrame
    __column_wise_mean_non_keyed: cp.ndarray
    __column_wise_normalisation_keyed: pd.DataFrame
    __column_wise_normalisation_non_keyed: cp.ndarray
    __column_wise_normalisation_computed: bool
    __column_wise_standard_deviation_keyed: pd.DataFrame
    __column_wise_standard_deviation_non_keyed: cp.ndarray
    __data_frame: pd.DataFrame
    __feature_names: list[str]
    __label_name: str
    __transposed_normalised_features_keyed: dict[str, cp.array]
    __transposed_normalised_features_non_keyed: cp.ndarray
    __transposed_normalised_label: cp.array
    __use_keyed_data: bool

    def __init__(
            self
            , data_frame: pd.DataFrame
            , feature_names: list[str]
            , label_name: str
            , use_keyed_data: bool = True):
        """Initialises the class.
        :param data_frame: The Pandas data frame for which to generate metadata.
        :param feature_names: The list of features, by name, in the data_frame.
        :param label_name: The name of the label in the data_frame.
        :param use_keyed_data: Indicates whether data should be keyed by feature, label, and parameter names. Keyed data
        is useful for debugging, education, and visualisation. Non-keyed data is useful for performance optimisation.
        """
        if data_frame.empty:
            raise ValueError(co.EXCEPTION_MESSAGE_NONE_OR_EMPTY_VALUE_FORMAT.format("data_frame"))
        if not feature_names:
            raise ValueError(co.EXCEPTION_MESSAGE_NONE_OR_EMPTY_VALUE_FORMAT.format("feature_names"))
        if not label_name:
            raise ValueError(co.EXCEPTION_MESSAGE_NONE_OR_EMPTY_VALUE_FORMAT.format("label_name"))

        for feature_name in feature_names:
            if feature_name not in data_frame.columns:
                raise ValueError(co.EXCEPTION_MESSAGE_NOT_IN_DATA_FRAME_FORMAT.format(feature_name))

        if label_name not in data_frame.columns:
            raise ValueError(co.EXCEPTION_MESSAGE_NOT_IN_DATA_FRAME_FORMAT.format(label_name))

        self.__active_columns_names = feature_names + [label_name]
        self.__column_wise_normalisation_keyed = pd.DataFrame()
        self.__column_wise_normalisation_non_keyed = cp.ndarray([])
        self.__column_wise_normalisation_computed = False
        self.__data_frame = data_frame[self.__active_columns_names]
        self.__feature_names = feature_names
        self.__label_name = label_name
        self.__transposed_normalised_features_keyed = {feature_name: cp.array([]) for feature_name in feature_names}
        self.__transposed_normalised_features_non_keyed = cp.ndarray([])
        self.__transposed_normalised_label = cp.array([])
        self.__use_keyed_data = use_keyed_data
        # Mean and standard deviation are precomputed for performance reasons.
        # We expect data in rows and columns, therefore we compute the mean and standard deviation column-wise (axis=0).
        self.__column_wise_mean_keyed = cp.mean(self.__data_frame, axis=0)
        self.__column_wise_mean_non_keyed = cp.mean(self.__data_frame.to_numpy(), axis=0)
        self.__column_wise_standard_deviation_keyed = cp.std(self.__data_frame, axis=0)
        self.__column_wise_standard_deviation_non_keyed = cp.std(self.__data_frame.to_numpy(), axis=0)

    def get_column_wise_mean(self) -> Union[pd.DataFrame, cp.ndarray]:
        """Returns the column-wise mean of the data_frame.
        :return: The column-wise mean of the data_frame.
        """
        if self.__use_keyed_data:
            return self.__column_wise_mean_keyed
        else:
            return self.__column_wise_mean_non_keyed

    def get_column_wise_normalisation(self) -> Union[pd.DataFrame, cp.ndarray]:
        """Returns the column-wise normalisation of the data_frame.
        :return: The column-wise normalisation of the data_frame.
        """
        if self.__use_keyed_data:
            return self.__column_wise_normalisation_keyed
        else:
            return self.__column_wise_normalisation_non_keyed

    def get_column_wise_normalisation_computed(self) -> bool:
        """Returns whether the column-wise normalisation has been computed.
        :return: True if the column-wise normalisation has been computed; otherwise, False.
        """
        return self.__column_wise_normalisation_computed

    def get_column_wise_standard_deviation(self) -> Union[pd.DataFrame, cp.ndarray]:
        """Returns the column-wise standard deviation of the data_frame.
        :return: The column-wise standard deviation of the data_frame.
        """
        if self.__use_keyed_data:
            return self.__column_wise_standard_deviation_keyed
        else:
            return self.__column_wise_standard_deviation_non_keyed

    def get_feature_count(self) -> int:
        """Returns the number of features in the data_frame.
        :return: The number of features in the data_frame.
        """
        return len(self.__feature_names)

    def get_label_count(self) -> int:
        """Returns the number of labels in the data_frame.
        :return: The number of labels in the data_frame.
        """
        return 1 if self.__label_name is not None else 0

    def get_transposed_normalised_features(self) -> Union[dict[str, cp.array], cp.ndarray]:
        """Returns the transposed normalised features."""
        if self.__use_keyed_data:
            return self.__transposed_normalised_features_keyed
        else:
            return self.__transposed_normalised_features_non_keyed

    def get_transposed_normalised_label(self) -> cp.array:
        """Returns the transposed normalised labels."""
        return self.__transposed_normalised_label

    def compute_column_wise_normalisation(self, force_recompute: bool = False):
        """Computes the column-wise normalisation for the dataset."""
        # Don't recompute if it's already been done.
        if self.__column_wise_normalisation_computed and not force_recompute:
            return

        if self.__use_keyed_data:
            self.__column_wise_normalisation_keyed = \
                ((self.__data_frame - self.__column_wise_mean_keyed)
                 / self.__column_wise_standard_deviation_keyed)
        else:
            self.__column_wise_normalisation_non_keyed = \
                ((self.__data_frame.to_numpy() - self.__column_wise_mean_non_keyed)
                 / self.__column_wise_standard_deviation_non_keyed)

        self.__column_wise_normalisation_computed = True

    def transpose_normalised_column_vectors(self):
        """Transposes the normalised column vectors."""
        if not self.__column_wise_normalisation_computed:
            self.compute_column_wise_normalisation()

        if self.__use_keyed_data:
            # Transpose the normalised label.
            norm = self.__column_wise_normalisation_keyed[self.__label_name]
            self.__transposed_normalised_label = cp.array(norm).reshape(1, len(norm))
            # Transpose the normalised features.
            for feature_name in self.__feature_names:
                norm = self.__column_wise_normalisation_keyed[feature_name]
                self.__transposed_normalised_features_keyed[feature_name] = cp.array(norm).reshape(1, len(norm))
        else:
            data_count = len(self.__data_frame)
            feature_count = len(self.__feature_names)
            variable_count = feature_count + 1
            self.__transposed_normalised_features_non_keyed = cp.ndarray((variable_count, data_count))
            for i in range(feature_count):
                self.__transposed_normalised_features_non_keyed[i] = cp.array(
                    self.__column_wise_normalisation_non_keyed[:, i])
            # Add a row of ones for the bias term, which facilitates matrix multiplication.
            self.__transposed_normalised_features_non_keyed[variable_count - 1] = cp.ones((1, data_count))
            self.__transposed_normalised_label = cp.array([self.__column_wise_normalisation_non_keyed[:, -1]])
