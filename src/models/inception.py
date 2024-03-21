import torch
from typing import Union
from collections import OrderedDict
from torch.nn import Linear, Sequential
from torchvision.models.googlenet import GoogLeNet
from torchvision.models.inception import Inception3
from torchvision.models import (
    GoogLeNet_Weights,
    googlenet,
    Inception_V3_Weights,
    inception_v3,
)
from prediction.predictor_model import ImageClassifier


def get_model(
    model_name: str,
    num_classes: int,
    pretrained=True,
    inference=False,
    dropout: float = 0.0,
) -> Union[GoogLeNet, Inception3]:
    """
    Retrieves a specified model by name, with the option to load it with pretrained weights.
    Adjusts the final fully connected layer to match the number of classes for the target task.

    Args:
    - model_name (str): Name of the model to retrieve ('inceptionV1' or 'inceptionV3').
    - num_classes (int): Number of classes for the final classification layer.
    - pretrained (bool): Whether to load the model with pretrained weights (default: True).
    - inference (bool): Indicates if the model is used for inference, affecting aux_logits (default: False).
    - dropout (float): Dropout rate for the fully connected layer (default: 0.0).

    Returns:
    - Union[GoogLeNet, Inception3]: The modified model instance.

    Raises:
    - ValueError: If an unsupported model name is provided.
    """
    models = {
        "inceptionV1": (GoogLeNet_Weights.IMAGENET1K_V1, googlenet),
        "inceptionV3": (Inception_V3_Weights.IMAGENET1K_V1, inception_v3),
    }

    requires_aux = not ((model_name == "inceptionV1") and inference)

    if model_name in models.keys():
        weights = models[model_name][0]
        model = models[model_name][1]
        model = (
            model(weights=weights, transform_input=False)
            if pretrained
            else model(pretrained=False, aux_logits=requires_aux)
        )
        in_features = model.fc.in_features
        if dropout > 0:
            model.fc = Sequential(
                torch.nn.Dropout(dropout),
                Linear(in_features, num_classes),
            )
        else:
            model.fc = Linear(in_features, num_classes)
        return model
    raise ValueError(f"Invalid model name. Supported models {models.keys()}")


class Inception(ImageClassifier):
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
        self.dropout = dropout
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
    def load(cls, params: dict, model_state: OrderedDict) -> "Inception":
        """
        Loads a pretrained model and its training configuration from a specified path.

        Args:
        - predictor_dir_path (str): Path to the directory with model's parameters and state.

        Returns:
        - Inception: A trainer object with the loaded model and training configuration.
        """
        model_name = params["model_name"]
        num_classes = params["num_classes"]
        model = get_model(
            model_name=model_name,
            num_classes=num_classes,
            pretrained=False,
            inference=True,
        )

        model.load_state_dict(model_state)

        trainer = Inception(**params)
        trainer.model = model
        return trainer
