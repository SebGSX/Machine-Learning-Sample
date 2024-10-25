# © 2024 Seb Garrioch. All rights reserved.
# Published under the MIT License.
from src.data import DataSetManager, DatasetMetadata

if __name__ == "__main__": # pragma: no cover
    manager: DataSetManager = DataSetManager()
    manager.acquire_kaggle_dataset("devzohaib/tvmarketingcsv", "tvmarketing.csv")
    dataset = manager.get_dataset(0)
    metadata: DatasetMetadata = DatasetMetadata(dataset, ["TV"], ["Sales"])
    metadata.compute_column_wise_normalisation()
    metadata.transpose_normalised_column_vectors()
