# © 2025 Seb Garrioch. All rights reserved.
# Published under the MIT License.

class ModelError(Exception):
    """Raised when there is an issue with the model."""

    __message: str = "An error occurred with the model."

    def __init__(self, message):
        self.__message = message
        super().__init__(self.__message)
