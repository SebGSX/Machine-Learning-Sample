# © 2024 Seb Garrioch. All rights reserved.
# Published under the MIT License.
import json

from pathlib import Path


class ConfigManager:
    """Manages the configuration files for the project."""

    __config_path: Path

    def __init__(self, config_path: str):
        """Initialises the ConfigManager object.
        :param config_path: The path to the config.json file containing the configuration settings.
        """
        self.__config_path = Path(config_path)

    def load_config(self) -> dict:
        """Loads the configuration settings from the config.json file.
        :return: The configuration settings as a dictionary.
        """
        try:
            if not self.__config_path.exists():
                raise FileNotFoundError(f"Config file not found at {self.__config_path}")
            with open(self.__config_path) as file:
                return json.load(file)
        except Exception as e:
            print(f"Error loading config file: {e}")
            raise
