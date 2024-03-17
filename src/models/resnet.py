import torch
from prediction.predictor_model import ImageClassifier
from torch.nn import Linear, ReLU, Sequential
from collections import OrderedDict

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


def get_model(model_name: str, num_classes: int, pretrained=True) -> torch.nn.Module:
    """
    Retrieves a specified ResNet model by name, optionally loading it with pretrained weights,
    and adjusts its fully connected layer to match the specified number of output classes.

    Args:
    - model_name (str): Name of the ResNet model to retrieve ('resnet18', 'resnet34', 'resnet50', 'resnet101', 'resnet152').
    - num_classes (int): Number of classes for the new fully connected layer.
    - pretrained (bool, optional): Whether to load the model with pretrained weights. Defaults to True.

    Returns:
    - torch.nn.Module: The modified ResNet model with the updated fully connected layer.

    Raises:
    - ValueError: If an unsupported model name is provided.
    """
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
        model.fc = Sequential[Linear(in_features, num_classes), ReLU()]
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
    def load(cls, params: dict, model_state: OrderedDict) -> "ResNet":
        """
        Loads a pretrained model and its training configuration from a specified path.

        Args:
        - predictor_dir_path (str): Path to the directory with model's parameters and state.

        Returns:
        - ResNet: A trainer object with the loaded model and training configuration.
        """
        model_name = params["model_name"]
        num_classes = params["num_classes"]
        model = get_model(
            model_name=model_name, num_classes=num_classes, pretrained=False
        )

        model.load_state_dict(model_state)

        trainer = ResNet(**params)
        trainer.model = model
        return trainer
