import os
import joblib
import numpy as np
from pathlib import Path
from typing import List, Tuple
from torchvision import transforms
from torchvision.datasets import ImageFolder
from torch.utils.data import DataLoader, Subset


class CustomDataLoader:
    def __init__(
        self,
        batch_size=64,
        num_workers=6,
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225],
        image_size=(224, 224),
        validation_size=0.0,
    ):
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.mean = mean
        self.std = std
        self.image_size = tuple(image_size)
        self.validation_size = validation_size
        self.transform = transforms.Compose(
            [
                transforms.Resize(self.image_size),
                transforms.ToTensor(),
                transforms.Normalize(mean=self.mean, std=self.std),
            ]
        )

    def create_data_loader(
        self,
        data_dir_path: str,
        create_validation: bool = False,
        val_size: float = 0.0,
        shuffle: bool = False,
    ):
        dataset = ImageFolder(root=data_dir_path, transform=self.transform)
        self.class_to_idx = dataset.class_to_idx
        self.num_classes = len(dataset.classes)
        self.class_names = dataset.classes
        if create_validation and val_size > 0:
            dataset, validation_dataset = (
                CustomDataLoader.stratified_split_to_dataloaders(
                    dataset, val_size=self.validation_size
                )
            )

        data_loader = DataLoader(
            dataset,
            batch_size=self.batch_size,
            shuffle=shuffle,
            num_workers=self.num_workers,
        )

        if create_validation:
            val_loader = DataLoader(
                validation_dataset,
                batch_size=self.batch_size,
                shuffle=False,
                num_workers=self.num_workers,
            )
            return data_loader, val_loader
        return data_loader

    def create_test_data_loader(
        self,
        data_dir_path: str,
    ) -> Tuple[DataLoader, List]:
        dataset = ImageFolder(root=data_dir_path, transform=self.transform)
        data_loader = DataLoader(
            dataset,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=self.num_workers,
        )
        image_names = [Path(i[0]).name for i in dataset.imgs]
        return data_loader, image_names

    @staticmethod
    def stratified_split_to_dataloaders(dataset: ImageFolder, val_size: float):
        targets = np.array([dataset.targets[i] for i in range(len(dataset))])
        classes, class_counts = np.unique(targets, return_counts=True)
        class_indices = [np.where(targets == i)[0] for i in classes]

        train_indices, val_indices = [], []

        for indices in class_indices:
            np.random.shuffle(indices)
            split = int(len(indices) * val_size)
            val_indices.extend(indices[:split])
            train_indices.extend(indices[split:])

        train_subset = Subset(dataset, train_indices)
        val_subset = Subset(dataset, val_indices)

        return train_subset, val_subset

    def save(self, file_path: str) -> None:
        path = Path(file_path)
        directory_path = path.parent
        os.makedirs(directory_path, exist_ok=True)
        joblib.dump(self, file_path)

    @staticmethod
    def load(file_path: str) -> None:
        return joblib.load(file_path)
