import torch
from prediction.predictor_model import ImageClassifier
from torch.nn import Linear, Sequential
from collections import OrderedDict

from torchvision.models import (
    VGG11_Weights,
    vgg11,
    VGG13_Weights,
    vgg13,
    VGG16_Weights,
    vgg16,
    VGG19_Weights,
    vgg19,
)


def get_model(
    model_name: str, num_classes: int, pretrained=True, dropout: float = 0.0
) -> torch.nn.Module:
    """
    Retrieves a specified VGG model by name, optionally loading it with pretrained weights,
    and adjusts its fully connected layer to match the specified number of output classes.

    Args:
    - model_name (str): Name of the ResNet model to retrieve ('vgg11', 'vgg13', 'vgg16', 'vgg19').
    - num_classes (int): Number of classes for the new fully connected layer.
    - pretrained (bool, optional): Whether to load the model with pretrained weights. Defaults to True.
    - dropout (float, optional): Dropout rate for the fully connected layer. Defaults to 0.0.

    Returns:
    - torch.nn.Module: The modified ResNet model with the updated fully connected layer.

    Raises:
    - ValueError: If an unsupported model name is provided.
    """
    models = {
        "vgg11": (VGG11_Weights.DEFAULT, vgg11),
        "vgg13": (VGG13_Weights.DEFAULT, vgg13),
        "vgg16": (VGG16_Weights.DEFAULT, vgg16),
        "vgg19": (VGG19_Weights.DEFAULT, vgg19),
    }

    if model_name in models.keys():
        weights = models[model_name][0]
        model = models[model_name][1]
        model = model(weights=weights) if pretrained else model(pretrained=False)
        in_features = model.classifier[6].in_features

        model.classifier[6] = Sequential(
            torch.nn.Dropout(dropout),
            Linear(in_features, num_classes),
        )
        return model
    raise ValueError(f"Invalid model name. Supported models {models.keys()}")


class VGG(ImageClassifier):
    def __init__(
        self,
        model_name: str,
        num_classes: int,
        lr: float = 0.01,
        dropout: float = 0.0,
        optimizer: str = "adam",
        max_epochs: int = 10,
        log_losses: str = "valid",
        early_stopping: bool = False,
        early_stopping_patience: int = 10,
        early_stopping_delta: float = 0.05,
        lr_scheduler: str = None,
        lr_scheduler_kwargs: dict = None,
        **kwargs,
    ):
        self.model_name = model_name
        self.droupout = dropout
        self.model = get_model(
            model_name=model_name, num_classes=num_classes, dropout=dropout
        )
        super().__init__(
            num_classes=num_classes,
            lr=lr,
            optimizer=optimizer,
            max_epochs=max_epochs,
            log_losses=log_losses,
            early_stopping=early_stopping,
            early_stopping_delta=early_stopping_delta,
            early_stopping_patience=early_stopping_patience,
            lr_scheduler=lr_scheduler,
            lr_scheduler_kwargs=lr_scheduler_kwargs,
            **kwargs,
        )

    @classmethod
    def load(cls, params: dict, model_state: OrderedDict) -> "VGG":
        """
        Loads a pretrained model and its training configuration from a specified path.

        Args:
        - predictor_dir_path (str): Path to the directory with model's parameters and state.

        Returns:
        - VGG: A trainer object with the loaded model and training configuration.
        """
        model_name = params["model_name"]
        num_classes = params["num_classes"]
        dropout = params.get("dropout", 0)
        model = get_model(
            model_name=model_name,
            num_classes=num_classes,
            pretrained=False,
            dropout=dropout,
        )

        model.load_state_dict(model_state)

        trainer = VGG(**params)
        trainer.model = model
        return trainer
