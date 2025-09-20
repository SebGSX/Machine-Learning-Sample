# © 2025 Seb Garrioch. All rights reserved.
# Published under the MIT License.
from src.config.config_manager import ConfigManager
from pytest_mock import MockerFixture


def test_load_config(mocker: MockerFixture):
    """Tests the load_config method.
    :param mocker: The mocker fixture.
    """
    mocked_exists = mocker.patch("pathlib.Path.exists", return_value=True)
    mocked_open = mocker.patch("builtins.open", mocker.mock_open(read_data='{"key": "value"}'))
    config_manager = ConfigManager("config/config.json")
    config = config_manager.load_config()
    assert config is not None
    mocked_exists.assert_called_once()
    mocked_open.assert_called_once()

def test_load_config_file_not_found(mocker: MockerFixture):
    """Tests the load_config method when the config file is not found.
    :param mocker: The mocker fixture.
    """
    mocked_exists = mocker.patch("pathlib.Path.exists", return_value=False)
    config_manager = ConfigManager("config/config.json")
    try:
        config_manager.load_config()
    except FileNotFoundError as e:
        assert str(e) == "Config file not found at config/config.json"
    mocked_exists.assert_called_once()
