# © 2024 Seb Garrioch. All rights reserved.
# Published under the MIT License.
from src.config.config_manager import ConfigManager


def test_initialisation_with_auth():
    """Tests the initialisation of the ConfigManager class with an auth file."""
    config_manager = ConfigManager("config/config.json", "config/auth-config.json")
    assert config_manager is not None

def test_initialisation_without_auth():
    """Tests the initialisation of the ConfigManager class without an auth file."""
    config_manager = ConfigManager("config/config.json")
    assert config_manager is not None

def test_load_config():
    """Tests the load_config method."""
    config_manager = ConfigManager("test-config.json")
    config = config_manager.load_config()
    assert config is not None
    assert "kaggle" in config
    assert "active_dataset" in config["kaggle"]
    assert config["kaggle"]["active_dataset"] == 0
    assert "datasets" in config["kaggle"]
    assert len(config["kaggle"]["datasets"]) > 0
    assert config["kaggle"]["datasets"][0]["file_name"] == "test-data.csv"
    assert config["kaggle"]["datasets"][0]["handle"] == "test/test-data"
    assert config["kaggle"]["datasets"][0]["label"] == "Label"
    assert config["kaggle"]["datasets"][0]["name"] == "test-data"
    assert "feature_names" in config["kaggle"]["datasets"][0]
    assert len(config["kaggle"]["datasets"][0]["feature_names"]) > 0
    assert config["kaggle"]["datasets"][0]["feature_names"][0] == "Feature"