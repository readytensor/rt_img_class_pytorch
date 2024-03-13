import os
import joblib
import torch
from prediction.predictor_model import ImageClassifier
from torch.nn import Linear
from torch.utils.data import DataLoader
from torchvision.models.resnet import ResNet as TorchResNet

from torchvision.models import (
    ResNet18_Weights,
    resnet18,
    ResNet34_Weights,
    resnet34,
    ResNet50_Weights,
    resnet50,
    ResNet101_Weights,
    resnet101,
    ResNet152_Weights,
    resnet152,
)


def get_model(model_name: str, num_classes: int, pretrained=True) -> TorchResNet:
    models = {
        "resnet18": (ResNet18_Weights, resnet18),
        "resnet34": (ResNet34_Weights, resnet34),
        "resnet50": (ResNet50_Weights, resnet50),
        "resnet101": (ResNet101_Weights, resnet101),
        "resnet152": (ResNet152_Weights, resnet152),
    }

    if model_name in models.keys():
        weights = models[model_name][0]
        model = models[model_name][1]
        model = model(weights=weights) if pretrained else model(pretrained=False)
        in_features = model.fc.in_features
        model.fc = Linear(in_features, num_classes)
        return model
    raise ValueError(f"Invalid model name. Supported models {models.keys()}")


class ResNet(ImageClassifier):
    def __init__(
        self,
        model_name: str,
        num_classes: int,
        lr: float = 0.01,
        optimizer: str = "adam",
        max_epochs: int = 10,
        early_stopping: bool = False,
        early_stopping_patience: int = 10,
        early_stopping_delta: float = 0.05,
        lr_scheduler: str = None,
        lr_scheduler_kwargs: dict = None,
        **kwargs,
    ):
        self.model_name = model_name
        self.model = get_model(model_name=model_name, num_classes=num_classes)
        super().__init__(
            num_classes=num_classes,
            lr=lr,
            optimizer=optimizer,
            max_epochs=max_epochs,
            early_stopping=early_stopping,
            early_stopping_delta=early_stopping_delta,
            early_stopping_patience=early_stopping_patience,
            lr_scheduler=lr_scheduler,
            lr_scheduler_kwargs=lr_scheduler_kwargs,
            **kwargs,
        )

    @classmethod
    def load(cls, predictor_dir_path: str) -> "ResNet":
        """
        Loads a pretrained model and its training configuration from a specified path.

        Args:
        - predictor_dir_path (str): Path to the directory with model's parameters and state.

        Returns:
        - ResNet: A trainer object with the loaded model and training configuration.
        """
        params_path = os.path.join(predictor_dir_path, "model_params.joblib")
        model_path = os.path.join(predictor_dir_path, "model_state.pth")
        params = joblib.load(params_path)
        model_state = torch.load(model_path)

        model_name = params["model_name"]
        num_classes = params["num_classes"]
        model = get_model(model_name=model_name, num_classes=num_classes)

        model.load_state_dict(model_state)

        trainer = ResNet(**params)
        trainer.model = model
        return trainer


def train_predictor_model(
    model_name: str,
    train_data: DataLoader,
    hyperparameters: dict,
    num_classes: int,
    valid_data: DataLoader = None,
) -> ImageClassifier:
    """
    Instantiate and train the classifier model.

    Args:
        train_data (DataLoader): The training data.
        hyperparameters (dict): Hyperparameters for the model.
        num_classes (int): Number of classes in the classificatiion problem.
        valid_data (DataLoader): The validation data.

    Returns:
        'ImageClassifier': The ImageClassifier model
    """

    model = ResNet(
        model_name=model_name,
        num_classes=num_classes,
        **hyperparameters,
    )
    model.fit(
        train_data=train_data,
        valid_data=valid_data,
    )
    return model


def load_predictor_model(predictor_dir_path: str) -> ResNet:
    """
    Load the ImageClassifier model from disk.

    Args:
        predictor_dir_path (str): Dir path where model is saved.

    Returns:
        ImageClassifier: A new instance of the loaded ImageClassifier model.
    """
    return ResNet.load(predictor_dir_path)
