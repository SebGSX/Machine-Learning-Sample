# © 2024 Seb Garrioch. All rights reserved.
# Published under the MIT License.
import cupy as cp
import os
import pandas as pd
import pytest

from collections import namedtuple
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

