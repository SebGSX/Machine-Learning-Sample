# © 2024 Seb Garrioch. All rights reserved.
# Published under the MIT License.
import cupy as cp
import pandas as pd

class DatasetMetadata:
    """Represents metadata about a dataset in tabular format. Columns are expected to represent features or labels while
    rows are expected to represent individual events, experiments, instances, observations, samples, etc."""

    __column_wise_mean: pd.DataFrame
    __column_wise_normalisation: pd.DataFrame
    __column_wise_normalisation_computed: bool
    __column_wise_standard_deviation: pd.DataFrame
    __data_frame: pd.DataFrame
    __feature_names: list[str]
    __label_names: list[str]
    __transposed_normalised_features: dict
    __transposed_normalised_labels: dict

    def __init__(self, data_frame: pd.DataFrame, feature_names: list[str], label_names: list[str]):
        """Initializes the class.
        :param data_frame: The Pandas data frame for which to generate metadata.
        :param feature_names: The list of features, by name, in the data_frame.
        :param label_names: The list of labels, by name, in the data_frame.
        """
        self.__column_wise_normalisation = pd.DataFrame()
        self.__column_wise_normalisation_computed = False
        self.__data_frame = data_frame
        self.__feature_names = feature_names
        self.__label_names = label_names
        self.__transposed_normalised_features: dict = {feature_name: cp.array([]) for feature_name in feature_names}
        self.__transposed_normalised_labels: dict = {label_name: cp.array([]) for label_name in label_names}
        # Mean and standard deviation are precomputed for performance reasons.
        # We expect data in rows and columns, therefore we compute the mean and standard deviation column-wise (axis=0).
        self.__column_wise_mean = cp.mean(self.__data_frame, axis=0)
        self.__column_wise_standard_deviation = cp.std(self.__data_frame, axis=0)

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

    def get_feature_count(self) -> int:
        """Returns the number of features in the data_frame.
        :return: The number of features in the data_frame.
        """
        return len(self.__feature_names)

    def get_label_count(self) -> int:
        """Returns the number of labels in the data_frame.
        :return: The number of labels in the data_frame.
        """
        return len(self.__label_names)

    def get_transposed_normalised_features(self) -> dict:
        """Returns the transposed normalised features."""
        return self.__transposed_normalised_features

    def get_transposed_normalised_labels(self) -> dict:
        """Returns the transposed normalised labels."""
        return self.__transposed_normalised_labels

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

        for feature in self.__feature_names:
            norm = self.__column_wise_normalisation[feature]
            self.__transposed_normalised_features[feature] = cp.array(norm).reshape(1, len(norm))

        for label in self.__label_names:
            norm = self.__column_wise_normalisation[label]
            self.__transposed_normalised_labels[label] = cp.array(norm).reshape(1, len(norm))
