import os
import warnings
import joblib
import numpy as np
from typing import Tuple

import torch
from torch.optim import Optimizer
from torch.nn import CrossEntropyLoss
from torch.utils.data import DataLoader
import torch.nn.functional as F

from torch_utils.early_stopping import EarlyStopping
from torch.optim.lr_scheduler import (
    ReduceLROnPlateau,
    CosineAnnealingLR,
    ExponentialLR,
    StepLR,
    _LRScheduler,
)
from torch_utils.lr_scheduler import WarmupCosineAnnealing
from logger import get_logger
from tqdm import tqdm


warnings.filterwarnings("ignore")

logger = get_logger(task_name="model")

# Check for GPU availability
device = "cuda:0" if torch.cuda.is_available() else "cpu"
logger.info(f"Using device: {device}")


def get_loss(model, data_loader, loss_function) -> float:
    model.eval()
    loss_total = 0
    with torch.no_grad():
        for data in data_loader:
            X, y = data[0].to(device), data[1].to(device)
            output = model(X)
            loss = loss_function(output, y)
            loss_total += loss.item()
    return loss_total / len(data_loader)


def get_optimizer(optimizer: str) -> Optimizer:
    supported_optimizers = {"adam": torch.optim.Adam, "sgd": torch.optim.SGD}

    if optimizer not in supported_optimizers.keys():
        raise ValueError(
            f"{optimizer} is not a supported optimizer. Supported: {supported_optimizers}"
        )
    return supported_optimizers[optimizer]


def get_lr_scheduler(scheduler: str) -> _LRScheduler:

    supported_schedulers = {
        "step": StepLR,
        "exponential": ExponentialLR,
        "plateau": ReduceLROnPlateau,
        "cosine_annealing": CosineAnnealingLR,
        "warmup_cosine_annealing": WarmupCosineAnnealing,
    }
    if scheduler not in supported_schedulers.keys():
        raise ValueError(
            f"{scheduler} is not a supported scheduler. Supported: {supported_schedulers}"
        )
    return supported_schedulers[scheduler]


class ImageClassifier:
    """
    This class provides a consistent interface that can be used with other
    image classifier models.
    """

    MODEL_NAME = "Image_Classifier"

    def __init__(
        self,
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
        """
        Construct a new ResNet18 image classifier

        Args:
        - num_classes (int): Number of output classes in the dataset.
        - lr (float): Learning rate for the optimizer. Default is 0.001.
        - optimizer (str): Name of the optimizer to use for training. Default is "adam". supported optimizers: {"adam", "sgd"}
        - max_epochs (int): Maximum number of training epochs. Default is 10.
        - early_stopping (bool): Whether to enable early stopping. Default is False.
        - early_stopping_patience (int): Number of epochs with no improvement after which training will be stopped. Default is 10.
        - early_stopping_delta (float): Minimum change in the monitored quantity to qualify as an improvement. Default is 0.05.
        - lr_scheduler (str): Name of the learning rate scheduler to use. If None, no scheduler will be used. Default is None.
        supported schedulers: {"step", "exponential", "plateau", "cosine_annealing"}
        - lr_scheduler_kwargs (dict): Keyword arguments to pass to the learning rate scheduler constructor. Default is None.

        Note:
        - The `lr_scheduler_kwargs` should contain any necessary arguments needed by the specified learning rate scheduler, excluding those arguments automatically determined by the training process, such as the optimizer.

        """
        self.lr = lr
        self.optimizer_str = optimizer
        self.max_epochs = max_epochs
        self.num_classes = num_classes
        self.early_stopping = early_stopping
        self.early_stopping_delta = early_stopping_delta
        self.early_stopping_patience = early_stopping_patience
        self.lr_scheduler_str = lr_scheduler
        self.lr_scheduler_kwargs = lr_scheduler_kwargs
        self.loss_function = CrossEntropyLoss()
        self.kwargs = kwargs

        self.optimizer = get_optimizer(optimizer)(self.model.parameters(), lr=lr)
        if self.lr_scheduler_str is not None:
            self.lr_scheduler = get_lr_scheduler(lr_scheduler)(
                self.optimizer, **self.lr_scheduler_kwargs
            )

        else:
            self.lr_scheduler = None

    def forward_backward(self, data: DataLoader):
        self.model.train()
        train_progress_bar = tqdm(
            total=len(data),
            desc="Epoch progress",
        )
        for inputs, labels in data:
            inputs, labels = inputs.to(device), labels.to(device)
            self.optimizer.zero_grad()
            outputs = self.model(inputs)
            if isinstance(outputs, tuple):
                outputs = outputs[0]

            loss = self.loss_function(outputs, labels)
            loss.backward()
            self.optimizer.step()

            train_progress_bar.update(1)
        train_progress_bar.close()

    def fit(self, train_data: DataLoader, valid_data: DataLoader = None):
        last_lr = self.lr
        self.model.to(device)
        early_stopper = EarlyStopping(
            patience=self.early_stopping_patience,
            delta=self.early_stopping_delta,
        )
        for epoch in range(self.max_epochs):

            self.forward_backward(train_data)

            if valid_data is not None:
                val_loss = get_loss(self.model, valid_data, self.loss_function)
                logger.info(f"Validation loss for epoch {epoch+1}: {val_loss:.3f}")
            else:
                train_loss = get_loss(self.model, train_data, self.loss_function)
                logger.info(f"Train loss for epoch {epoch+1}: {train_loss:.3f}")

            if self.lr_scheduler is not None:
                loss = val_loss if valid_data is not None else train_loss
                self.lr_scheduler.step(loss)
                scheduler_lr = self.lr_scheduler.get_last_lr()[0]
                if last_lr != scheduler_lr:
                    logger.info(f"Learning rate set to {scheduler_lr}")
                    last_lr = scheduler_lr

            if self.early_stopping:
                loss = val_loss if valid_data is not None else train_loss
                if early_stopper(loss):
                    logger.info(f"Early stopping after {epoch+1} epochs")
                    break

    def predict(self, data: DataLoader) -> Tuple[np.ndarray, np.ndarray]:
        """
        Predicts the class labels and probabilities for the given data.

        Args:
            data (DataLoader): The input data.

        Returns:
            Tuple[np.ndarray, np.ndarray]: (Predicted class labels, Probabilities).
        """
        self.model.eval()
        self.model.to(device)
        with torch.no_grad():
            all_labels, all_predicted, all_probs = (
                np.array([]),
                np.array([]),
                np.array([]),
            )

            for inputs, labels in data:
                inputs, labels = inputs.to(device), labels.to(device)
                outputs = self.model(inputs)
                probs = F.softmax(outputs, dim=1)
                _, predicted = torch.max(outputs.data, 1)

                # Convert tensors to numpy arrays before appending
                all_predicted = np.append(all_predicted, predicted.cpu().numpy())
                all_labels = np.append(all_labels, labels.cpu().numpy())
                all_probs = (
                    np.concatenate((all_probs, probs.cpu().numpy()), axis=0)
                    if all_probs.size
                    else probs.cpu().numpy()
                )

        return all_predicted, all_probs

    def save(self, predictor_dir_path: str) -> None:
        """
        Saves the model's state dictionary and training parameters to the specified path.

        This method saves two files:
        one with the model's parameters (such as learning rate, number of classes, etc.
        and another with the model's state dictionary. The parameters are
        saved in a joblib file, and the model's state is saved in a PyTorch file.

        Args:
        - predictor_path (str): The directory path where the model parameters
          and state are to be saved.
        """
        model_params = {
            "model_name": self.model_name,
            "lr": self.lr,
            "optimizer": self.optimizer_str,
            "lr_scheduler": self.lr_scheduler_str,
            "lr_scheduler_kwargs": self.lr_scheduler_kwargs,
            "max_epochs": self.max_epochs,
            "early_stopping": self.early_stopping,
            "early_stopping_patience": self.early_stopping_patience,
            "early_stopping_delta": self.early_stopping_delta,
            "num_classes": self.num_classes,
        }
        params_path = os.path.join(predictor_dir_path, "model_params.joblib")
        model_path = os.path.join(predictor_dir_path, "model_state.pth")
        joblib.dump(model_params, params_path)
        torch.save(self.model.state_dict(), model_path)

    @classmethod
    def load(cls, predictor_dir_path: str) -> "ImageClassifier":
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

        if model_name.startswith("resnet"):
            from models.resnet import ResNet

            return ResNet.load(params, model_state)

        if model_name.startswith("inception"):
            from models.inception import Inception

            return Inception.load(params, model_state)

        if model_name.startswith("mnasnet"):
            from models.mnasnet import MNASNet

            return MNASNet.load(params, model_state)

    def evaluate(self, test_data: DataLoader):
        """Evaluate the model and return the loss"""
        return get_loss(
            self.model, data_loader=test_data, loss_function=self.loss_function
        )

    def __str__(self):
        # sort params alphabetically for unit test to run successfully
        return f"Model name: {self.MODEL_NAME}"


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

    if model_name.startswith("resnet"):
        from models.resnet import ResNet

        constructor = ResNet

    elif model_name.startswith("inception"):
        from models.inception import Inception

        constructor = Inception

    elif model_name.startswith("mnasnet"):
        from models.mnasnet import MNASNet

        constructor = MNASNet

    model = constructor(
        model_name=model_name,
        num_classes=num_classes,
        **hyperparameters,
    )
    model.fit(
        train_data=train_data,
        valid_data=valid_data,
    )
    return model


def predict_with_model(
    model: ImageClassifier, test_data: DataLoader
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Make predictions.

    Args:
        model (ImageClassifier): The ImageClassifier model.
        test_data (DataLoader): The test input data for model.

    Returns:
        Tuple[np.ndarray, np.ndarray]: (predicted class labels, predicted class probabilites).
    """
    labels, probabilites = model.predict(test_data)
    return labels, probabilites


def save_predictor_model(model: ImageClassifier, predictor_dir_path: str) -> None:
    """
    Save the ImageClassifier model to disk.

    Args:
        model (ImageClassifier): The Classifier model to save.
        predictor_dir_path (str): Dir path to which to save the model.
    """
    if not os.path.exists(predictor_dir_path):
        os.makedirs(predictor_dir_path)
    model.save(predictor_dir_path)


def load_predictor_model(predictor_dir_path: str) -> ImageClassifier:
    """
    Load the ImageClassifier model from disk.

    Args:
        predictor_dir_path (str): Dir path where model is saved.

    Returns:
        ImageClassifier: A new instance of the loaded ImageClassifier model.
    """

    return ImageClassifier.load(predictor_dir_path)


def evaluate_predictor_model(model: ImageClassifier, test_data: DataLoader) -> float:
    """
    Evaluate the ImageClassifier model and return the loss.

    Args:
        model (ImageClassifier): The Classifier model.
        test_data (DataLoader): The dataset to be evaluate the model on.

    Returns:
        float: The computed loss on the dataset.
    """
    return model.evaluate(test_data)
