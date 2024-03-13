from abc import ABC, abstractmethod
from typing import Any, Tuple


class AbstractDataLoaderFactory(ABC):
    @abstractmethod
    def create_train_and_valid_data_loaders(
        self, train_data_dir_path: str, validation_dir_path: str = None
    ) -> Tuple[Any, Any]:
        """
        Create a data loader for the specified dataset, with options for stratified splitting.

        Args:
            train_data_dir_path: The path to the training dataset directory.
            validation_dir_path: Optional; the path to the validation dataset directory.
                                If provided, the validation dataset is loaded from this
                                directory. Otherwise, a validation split can be created
                                from the training dataset.

        Returns:
            A tuple of data loader objects for the training and validation datasets.
            The validation data loader is None if no validation dataset is provided or
            created.
        """
        raise NotImplementedError

    @abstractmethod
    def create_test_data_loader(self, data_dir_path: str) -> Any:
        """
        Create a data loader for test data.

        Args:
            data_dir_path: Path to the test dataset directory.

        Returns:
            A data loader object for test data.
        """
        raise NotImplementedError
