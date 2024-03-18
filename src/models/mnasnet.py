import torch
from collections import OrderedDict
from prediction.predictor_model import ImageClassifier

from torchvision.models import (
    MNASNet0_5_Weights,
    mnasnet0_5,
    MNASNet1_0_Weights,
    mnasnet1_0,
    MNASNet1_3_Weights,
    mnasnet1_3,
)


def get_model(model_name: str, num_classes: int, pretrained=True) -> torch.nn.Module:
    """
    Retrieves a specified MNASNet model by name, optionally loading it with pretrained weights,
    and adjusts its classifier for a given number of output classes.

    Args:
    - model_name (str): Name of the MNASNet model to retrieve ('mnasnet0_5', 'mnasnet1_0', 'mnasnet1_3').
    - num_classes (int): Number of classes for the final classification layer.
    - pretrained (bool, optional): Whether to load the model with pretrained weights. Defaults to True.

    Returns:
    - torch.nn.Module: The modified MNASNet model with the updated classifier.

    Raises:
    - ValueError: If an unsupported model name is provided.
    """
    models = {
        "mnasnet0_5": (MNASNet0_5_Weights.IMAGENET1K_V1, mnasnet0_5),
        "mnasnet1_0": (MNASNet1_0_Weights.IMAGENET1K_V1, mnasnet1_0),
        "mnasnet1_3": (MNASNet1_3_Weights.IMAGENET1K_V1, mnasnet1_3),
    }

    if model_name in models.keys():
        weights = models[model_name][0]
        model = models[model_name][1]
        model = model(weights=weights) if pretrained else model(pretrained=False)
        model.classifier = torch.nn.Sequential(
            torch.nn.Dropout(p=0.2),
            torch.nn.Linear(model.classifier[1].in_features, num_classes),
        )
        return model
    raise ValueError(f"Invalid model name. Supported models {models.keys()}")


class MNASNet(ImageClassifier):
    def __init__(
        self,
        model_name: str,
        num_classes: int,
        lr: float = 0.01,
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
        self.model = get_model(model_name=model_name, num_classes=num_classes)
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
    def load(cls, params: dict, model_state: OrderedDict) -> "MNASNet":
        """
        Loads a pretrained model and its training configuration from a specified path.

        Args:
        - predictor_dir_path (str): Path to the directory with model's parameters and state.

        Returns:
        - MNASNet: A trainer object with the loaded model and training configuration.
        """
        model_name = params["model_name"]
        num_classes = params["num_classes"]
        model = get_model(
            model_name=model_name, num_classes=num_classes, pretrained=False
        )
        model.load_state_dict(model_state)

        trainer = MNASNet(**params)
        trainer.model = model
        return trainer
