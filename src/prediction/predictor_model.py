import os
import warnings

import joblib
import numpy as np
from typing import Tuple

import torch
from torch.optim import Optimizer
from torch.nn import Linear, CrossEntropyLoss
from torch.utils.data import DataLoader
import torch.nn.functional as F
from torchvision.models import ResNet18_Weights, resnet18
from torch_utils.early_stopping import EarlyStopping
from logger import get_logger
from tqdm import tqdm

warnings.filterwarnings("ignore")

logger = get_logger(task_name="model")

# Check for GPU availability
device = "cuda:0" if torch.cuda.is_available() else "cpu"


def get_loss(model, data_loader, loss_function):
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


class ImageClassifier:
    """CNN Timeseries Forecaster.

    This class provides a consistent interface that can be used with other
    Forecaster models.
    """

    MODEL_NAME = "ResNet18_Image_Classifier"

    def __init__(
        self,
        num_classes: int,
        lr: float = 0.001,
        optimizer: str = "adam",
        max_epochs: int = 10,
        early_stopping: bool = True,
        early_stopping_patience: int = 10,
        early_stopping_delta: float = 0.05,
        **kwargs,
    ):
        """Construct a new ResNet18 image classifier."""
        self.lr = lr
        self.optimizer_str = optimizer
        self.max_epochs = max_epochs
        self.num_classes = num_classes
        self.early_stopping = early_stopping
        self.early_stopping_delta = early_stopping_delta
        self.early_stopping_patience = early_stopping_patience
        self.loss_function = CrossEntropyLoss()
        self.kwargs = kwargs

        model = resnet18(weights=ResNet18_Weights)
        in_features = model.fc.in_features
        model.fc = Linear(in_features, num_classes)
        self.model = model

        self.optimizer = get_optimizer(optimizer)(self.model.parameters(), lr=lr)

    def forward_backward(self, data: DataLoader):
        for inputs, labels in data:
            inputs, labels = inputs.to(device), labels.to(device)
            self.optimizer.zero_grad()
            outputs = self.model(inputs)
            if isinstance(outputs, tuple):
                outputs = outputs[0]

            # _, predicted = torch.max(outputs.data, 1)
            loss = self.loss_function(outputs, labels)
            loss.backward()
            self.optimizer.step()

    def fit(self, train_data: DataLoader, valid_data: DataLoader = None):
        self.model.to(device)
        self.model.train()
        total_batches = len(train_data)
        early_stopper = EarlyStopping(
            patience=self.early_stopping_patience,
            delta=self.early_stopping_delta,
            trace_func=logger.info,
        )
        for epoch in range(self.max_epochs):
            train_loss, val_loss = 0, 0
            train_progress_bar = tqdm(
                total=total_batches,
                desc=f"Training - Epoch {epoch + 1}/{self.max_epochs}",
            )

            self.forward_backward(train_data)

            train_loss += get_loss(self.model, train_data, self.loss_function)

            if valid_data is not None:
                val_loss += get_loss(self.model, valid_data, self.loss_function)

            train_progress_bar.update(1)

            if self.early_stopping:
                loss = val_loss if valid_data is not None else train_loss
                if early_stopper(loss):
                    break
        train_progress_bar.close()

    def predict(self, data: DataLoader) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        Predicts the class labels and logits for the data in a data loader.

        Args:
            data_loader (DataLoader): The input data.

        Returns:
            Tuple[np.ndarray, np.ndarray, np.ndarray]: (Truth labels, Predicted class labels, Probabilities).
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

        return all_labels, all_predicted, all_probs

    def summary(self):
        self.model.summary()

    def evaluate(self, test_data: DataLoader):
        """Evaluate the model and return the loss and metrics"""
        return get_loss(
            self.model, data_loader=test_data, loss_function=self.loss_function
        )

    def save(self, predictor_dir_path: str) -> None:
        """
        Saves the model's state dictionary and training parameters to the specified path.

        This method creates the directory if it doesn't exist and saves two files: one with
        the model's parameters (such as data loaders, number of classes, model name, and
        output folder) and another with the model's state dictionary. The parameters are
        saved in a joblib file, and the model's state is saved in a PyTorch file.

        Args:
        - predictor_path (str, optional): The directory path where the model parameters
          and state are to be saved.
        """
        model_params = {
            "lr": self.lr,
            "optimizer": self.optimizer_str,
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

    @staticmethod
    def load(predictor_dir_path: str) -> "ImageClassifier":
        """
        Loads a pretrained model and its training configuration from a specified path.

        Args:
        - predictor_dir_path (str): Path to the directory with model's parameters and state.

        Returns:
        - ImageClassifier: A trainer object with the loaded model and training configuration.
        """
        params_path = os.path.join(predictor_dir_path, "model_params.joblib")
        model_path = os.path.join(predictor_dir_path, "model_state.pth")
        params = joblib.load(params_path)
        model_state = torch.load(model_path)

        num_classes = params["num_classes"]
        model = resnet18(weights=ResNet18_Weights)

        in_features = model.fc.in_features
        model.fc = Linear(in_features, num_classes)

        model.load_state_dict(model_state)

        trainer = ImageClassifier(**params)
        trainer.model = model
        return trainer

    def __str__(self):
        # sort params alphabetically for unit test to run successfully
        return f"Model name: {self.MODEL_NAME}"


def train_predictor_model(
    train_data: DataLoader,
    hyperparameters: dict,
    num_classes: int,
    valid_data: DataLoader = None,
) -> ImageClassifier:
    """
    Instantiate and train the forecaster model.

    Args:
        history (np.ndarray): The training data inputs.
        forecast_length (int): Length of forecast window.
        hyperparameters (dict): Hyperparameters for the Forecaster.

    Returns:
        'Forecaster': The Forecaster model
    """
    model = ImageClassifier(
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
    _, labels, probabilites = model.predict(test_data)
    return labels, probabilites


def save_predictor_model(model: ImageClassifier, predictor_dir_path: str) -> None:
    """
    Save the Forecaster model to disk.

    Args:
        model (ImageClassifier): The Classifier model to save.
        predictor_dir_path (str): Dir path to which to save the model.
    """
    if not os.path.exists(predictor_dir_path):
        os.makedirs(predictor_dir_path)
    model.save(predictor_dir_path)


def load_predictor_model(predictor_dir_path: str) -> ImageClassifier:
    """
    Load the Forecaster model from disk.

    Args:
        predictor_dir_path (str): Dir path where model is saved.

    Returns:
        Forecaster: A new instance of the loaded Forecaster model.
    """
    return ImageClassifier.load(predictor_dir_path)


def evaluate_predictor_model(model: ImageClassifier, test_data: DataLoader) -> float:
    """
    Evaluate the Forecaster model and return the accuracy.

    Args:
        model (ImageClassifier): The Classifier model.
        test_data (DataLoader): The dataset to be evaluate the model on.

    Returns:
        float: The computed loss on the dataset.
    """
    return model.evaluate(test_data)
