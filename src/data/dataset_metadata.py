# © 2024 Seb Garrioch. All rights reserved.
# Published under the MIT License.
import src.common as co
import cupy as cp
import pandas as pd


class DatasetMetadata:
    """Represents metadata about a dataset in tabular format. Columns are expected to represent features or labels while
    rows are expected to represent individual events, experiments, instances, observations, samples, etc."""

    __active_columns_names: list[str]
    __column_wise_mean: pd.DataFrame
    __column_wise_normalisation: pd.DataFrame
    __column_wise_normalisation_computed: bool
    __column_wise_standard_deviation: pd.DataFrame
    __data_frame: pd.DataFrame
    __feature_names: list[str]
    __label_name: str
    __transposed_normalised_features: dict[str, cp.array]
    __transposed_normalised_label: cp.array

    def __init__(self, data_frame: pd.DataFrame, feature_names: list[str], label_name: str):
        """Initializes the class.
        :param data_frame: The Pandas data frame for which to generate metadata.
        :param feature_names: The list of features, by name, in the data_frame.
        :param label_name: The name of the label in the data_frame.
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
        self.__column_wise_normalisation = pd.DataFrame()
        self.__column_wise_normalisation_computed = False
        self.__data_frame = data_frame[self.__active_columns_names]
        self.__feature_names = feature_names
        self.__label_name = label_name
        self.__transposed_normalised_features = {feature_name: cp.array([]) for feature_name in feature_names}
        self.__transposed_normalised_label = cp.array([])
        # Mean and standard deviation are precomputed for performance reasons.
        # We expect data in rows and columns, therefore we compute the mean and standard deviation column-wise (axis=0).
        self.__column_wise_mean = cp.mean(self.__data_frame, axis=0)
        self.__column_wise_standard_deviation = cp.std(self.__data_frame, axis=0)

    def get_column_wise_mean(self) -> pd.DataFrame:
        """Returns the column-wise mean of the data_frame.
        :return: The column-wise mean of the data_frame.
        """
        return self.__column_wise_mean

    def get_column_wise_normalisation(self) -> pd.DataFrame:
        """Returns the column-wise normalisation of the data_frame.
        :return: The column-wise normalisation of the data_frame.
        """
        return self.__column_wise_normalisation

    def get_column_wise_normalisation_computed(self) -> bool:
        """Returns whether the column-wise normalisation has been computed.
        :return: True if the column-wise normalisation has been computed; otherwise, False.
        """
        return self.__column_wise_normalisation_computed

    def get_column_wise_standard_deviation(self) -> pd.DataFrame:
        """Returns the column-wise standard deviation of the data_frame.
        :return: The column-wise standard deviation of the data_frame.
        """
        return self.__column_wise_standard_deviation

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

    def get_transposed_normalised_features(self) -> dict[str, cp.array]:
        """Returns the transposed normalised features."""
        return self.__transposed_normalised_features

    def get_transposed_normalised_label(self) -> cp.array:
        """Returns the transposed normalised labels."""
        return self.__transposed_normalised_label

    def compute_column_wise_normalisation(self, force_recompute: bool = False):
        """Computes the column-wise normalisation for the dataset."""
        # Don't recompute if it's already been done.
        if self.__column_wise_normalisation_computed and not force_recompute:
            return

        self.__column_wise_normalisation = \
            ((self.__data_frame - self.__column_wise_mean) / self.__column_wise_standard_deviation)

        self.__column_wise_normalisation_computed = True

    def transpose_normalised_column_vectors(self):
        """Transposes the normalised column vectors."""
        if not self.__column_wise_normalisation_computed:
            self.compute_column_wise_normalisation()

        for feature_name in self.__feature_names:
            norm = self.__column_wise_normalisation[feature_name]
            self.__transposed_normalised_features[feature_name] = cp.array(norm).reshape(1, len(norm))

        norm = self.__column_wise_normalisation[self.__label_name]
        self.__transposed_normalised_label = cp.array(norm).reshape(1, len(norm))
