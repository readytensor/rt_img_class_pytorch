import os
import torch
import joblib
from typing import Tuple, Union
from pathlib import Path
import numpy as np
from torch.utils.data import DataLoader, Subset
from torchvision.datasets import ImageFolder
from torchvision import transforms

from data_loader.base_loader import AbstractDataLoaderFactory


class CustomImageFolder(ImageFolder):
    def __getitem__(self, index):
        # Call the parent class's __getitem__ to retrieve image and label
        original_tuple = super(CustomImageFolder, self).__getitem__(index)
        # Retrieve the path from self.imgs, which stores tuples of (path, class_index)
        path, _ = self.imgs[index]
        # Return a tuple with the filename, image, and label
        # Ensure path is a string, as expected by Path
        return (Path(path).name, *original_tuple)


class PyTorchDataLoaderFactory(AbstractDataLoaderFactory):
    def __init__(
        self,
        batch_size: int,
        num_workers: int,
        transforms: transforms.Compose,
        validation_size: float = 0.0,
        augmentation: bool = False,
        shuffle_train=True,
        random_state: int = 42,
    ):
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.validation_size = validation_size
        self.augmentation = augmentation
        self.transform = transforms
        self.num_classes = None
        self.shuffle_train = shuffle_train
        self.random_state = random_state
        self.num_classes = None

    def custom_collate_fn(self, batch):
        """
        Custom collate function to handle optional on-the-fly data augmentation.
        """
        augmentation_transforms = transforms.Compose(
            [
                transforms.RandomHorizontalFlip(p=0.5),
                transforms.RandomVerticalFlip(p=0.5),
                transforms.ColorJitter(
                    brightness=0.2, contrast=0.2, saturation=0.2, hue=0.1
                ),
                transforms.RandomRotation(degrees=(0, 360)),
            ]
        )
        augmented_batch = []
        for path, image, label in batch:
            augmented_batch.append((path, image, label))  # Original image
            if self.augmentation:
                # Apply augmentation and append
                augmented_image = augmentation_transforms(image)
                augmented_batch.append((path, augmented_image, label))
        # Unzip the batch items
        paths, images, labels = zip(*augmented_batch)
        return paths, torch.stack(images), torch.tensor(labels)

    def create_train_and_valid_data_loaders(
        self,
        train_dir_path: str,
        validation_dir_path: str = None,
    ) -> Tuple[DataLoader, Union[DataLoader, None]]:
        """
        Creates DataLoader objects for training and, if specified, validation datasets.

        This method automatically applies predefined transformations, encapsulates the
        datasets into DataLoader objects for batched processing, and determines whether
        to generate a validation split. A validation DataLoader is created either from
        a provided validation dataset directory or by splitting the training dataset,
        depending on the factory's configuration and the arguments passed.

        Args:
            train_data_dir_path: The path to the training dataset directory.
            validation_dir_path: Optional; the path to the validation dataset directory.
                                If provided, the validation dataset is loaded from this
                                directory. Otherwise, a validation split can be created
                                from the training dataset.
        Returns:
            A tuple of DataLoaders for the training and validation datasets.
            The validation DataLoader is None if no validation dataset is provided or
            created.
        """

        dataset = CustomImageFolder(root=train_dir_path, transform=self.transform)
        self.class_to_idx = dataset.class_to_idx
        self.num_classes = len(dataset.classes)
        self.class_names = dataset.classes

        idx_to_class = {v: k for k, v in self.class_to_idx.items()}

        # if validation data is given to us directly then load it
        if validation_dir_path is not None:
            validation_dataset = CustomImageFolder(
                root=validation_dir_path, transform=self.transform
            )
            val_loader = DataLoader(
                validation_dataset,
                batch_size=self.batch_size,
                shuffle=False,
                num_workers=self.num_workers,
            )
            train_loader = DataLoader(
                dataset,
                batch_size=self.batch_size,
                shuffle=self.shuffle_train,
                num_workers=self.num_workers,
                collate_fn=self.custom_collate_fn if self.augmentation else None,
            )
            self.train_image_names = [Path(i[0]).name for i in dataset.imgs]
            self.val_image_names = [Path(i[0]).name for i in validation_dataset.imgs]
            self.train_image_labels = [i[1] for i in dataset.imgs]
            self.val_image_labels = [i[1] for i in validation_dataset.imgs]

        else:
            if self.validation_size > 0:
                # Create validation split out of train split

                # Manual implementation of stratified sampling
                np.random.seed(self.random_state)
                targets = np.array(dataset.targets)
                classes, _ = np.unique(targets, return_counts=True)
                class_indices = [np.where(targets == i)[0] for i in classes]

                train_indices, val_indices = [], []
                for indices in class_indices:
                    np.random.shuffle(indices)
                    split = int(len(indices) * self.validation_size)
                    val_indices.extend(indices[:split])
                    train_indices.extend(indices[split:])

                train_subset = Subset(dataset, train_indices)
                val_subset = Subset(dataset, val_indices)

                train_loader = DataLoader(
                    train_subset,
                    batch_size=self.batch_size,
                    shuffle=self.shuffle_train,
                    num_workers=self.num_workers,
                    collate_fn=self.custom_collate_fn if self.augmentation else None,
                )
                val_loader = DataLoader(
                    val_subset,
                    batch_size=self.batch_size,
                    shuffle=False,
                    num_workers=self.num_workers,
                )
                self.train_image_names = [
                    Path(dataset.imgs[i][0]).name for i in train_indices
                ]
                self.val_image_names = [
                    Path(dataset.imgs[i][0]).name for i in val_indices
                ]
                self.train_image_labels = [dataset.imgs[i][1] for i in train_indices]
                self.val_image_labels = [dataset.imgs[i][1] for i in val_indices]
            else:
                # No validation data to use
                val_loader = None
                train_loader = DataLoader(
                    dataset,
                    batch_size=self.batch_size,
                    shuffle=self.shuffle_train,
                    num_workers=self.num_workers,
                    collate_fn=self.custom_collate_fn if self.augmentation else None,
                )
                self.train_image_names = [Path(i[0]).name for i in dataset.imgs]
                self.val_image_names = None
                self.train_image_labels = [i[1] for i in dataset.imgs]
                self.val_image_labels = None

        self.train_image_labels = [idx_to_class[i] for i in self.train_image_labels]
        if self.val_image_labels is not None:
            self.val_image_labels = [idx_to_class[i] for i in self.val_image_labels]
        return train_loader, val_loader

    def create_test_data_loader(self, data_dir_path: str):
        """
        Create a PyTorch DataLoader for test data.

        Args:
            data_dir_path: Path to the test dataset directory.

        Returns:
            A DataLoader for test data.
        """
        test_dataset = CustomImageFolder(root=data_dir_path, transform=self.transform)
        test_loader = DataLoader(
            test_dataset,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=self.num_workers,
        )
        image_names = [Path(i[0]).name for i in test_dataset.imgs]
        return test_loader, image_names

    def save(self, file_path: str) -> None:
        """
        Save the data loader factory to a file.

        Args:
            file_path (str): The path to the file where the data loader factory will
                             be saved.
        """
        path = Path(file_path)
        directory_path = path.parent
        os.makedirs(directory_path, exist_ok=True)
        joblib.dump(self, file_path)


def get_data_loader(model_name: str) -> PyTorchDataLoaderFactory:
    ordinary = {
        "resnet18",
        "resnet34",
        "resnet50",
        "resnet101",
        "resnet152",
        "inceptionV1",
        "mnasnet1_0",
        "mnasnet1_3",
        "mnasnet0_5",
        "vgg11",
        "vgg13",
        "vgg16",
        "vgg19",
    }
    inception = {"inceptionV3"}
    supported = ordinary | inception
    if model_name in ordinary:
        return OrdinaryDataLoader
    if model_name in inception:
        return InceptionV3DataLoader

    raise ValueError(f"Invalid model name. supported model names: {supported}")


def load_data_loader_factory(data_loader_file_path: str) -> PyTorchDataLoaderFactory:
    """
    Load the data loader factory from a file.
    """
    return joblib.load(data_loader_file_path)


class OrdinaryDataLoader(PyTorchDataLoaderFactory):
    TRANSFORMS = transforms.Compose(
        [
            transforms.Resize(256),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ]
    )

    def __init__(
        self,
        batch_size: int = 64,
        validation_size: float = 0.0,
        shuffle_train=True,
        num_workers: int = 0,
        random_state: int = 42,
        **kwargs,
    ):
        super().__init__(
            batch_size=batch_size,
            num_workers=num_workers,
            transforms=self.TRANSFORMS,
            validation_size=validation_size,
            shuffle_train=shuffle_train,
            random_state=random_state,
            **kwargs,
        )


class InceptionV3DataLoader(PyTorchDataLoaderFactory):
    TRANSFORMS = transforms.Compose(
        [
            transforms.Resize(299),
            transforms.CenterCrop(299),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ]
    )

    def __init__(
        self,
        batch_size: int = 64,
        validation_size: float = 0.0,
        shuffle_train=True,
        num_workers: int = 0,
        random_state: int = 42,
        **kwargs,
    ):
        super().__init__(
            batch_size=batch_size,
            num_workers=num_workers,
            transforms=self.TRANSFORMS,
            validation_size=validation_size,
            shuffle_train=shuffle_train,
            random_state=random_state,
            **kwargs,
        )
