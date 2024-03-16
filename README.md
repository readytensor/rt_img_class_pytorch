# Image Classifier with Pre-Training in PyTorch

Image Classifier for the Image Classification problem category as per Ready Tensor specifications.

## Project Description

Leveraging pre-trained models for image classification, this repository extends support to multiple architectures, including various ResNet versions, InceptionV1, InceptionV3, and MNASNet models. It's designed for ease of use and deployment, with minimal setup required for adapting to new datasets.


The following are the requirements for using your data with this model:

- The training and testing data must consist of multiple directories. Each directory represents a class in the classification problem with the relevant images inside the directory. See **/examples/mini_mnist.zip**
- Validation data are optional. Including a **validation** folder is possible. Alternatively, you can choose to create a validation set out of the training set by passing the validation set size parameter.
- The train and test files across all different class directories must have unique names. The class directories must be named after the label of that class.
- No preprocessing is required for the images. Necessary preprocessing steps will be applied for the data to match the format expected by the pre-trained model.
---

Here are the highlights of this implementation: <br/>

- **Supported models:** ResNet (18, 34, 50, 101, 152), InceptionV1, InceptionV3, MNASNet (0.5, 1.0, 1.3)  using **PyTorch**
  Additionally, the implementation contains the following features:
- **Data Preprocessing**: Preprocessinig pipeline using **torchvision**.
- **Error handling and logging**: Python's logging module is used for logging and key functions include exception handling.

## Project Structure

The following is the directory structure of the project:

- **`examples/`**: This directory contains example files for the mini-mnist dataset. Three files are included: `mini_mnist.zip`. You can extract the folders and place the training/testing folders in `model_inputs_outputs/inputs`.
- **`model_inputs_outputs/`**: This directory contains files that are either inputs to, or outputs from, the model. When running the model locally (i.e. without using docker), this directory is used for model inputs and outputs. This directory is further divided into:
  - **`/inputs/`**: This directory contains all the input files for this project, including the `training`, `testing` and optionally `validation` folders.
  - **`/model/artifacts/`**: This directory is used to store the model artifacts, such as trained models and their parameters.
  - **`/outputs/`**: The outputs directory contains sub-directories for error logs and prediction results.
- **`src/`**: This directory holds the source code for the project. It is further divided into various subdirectories:
  - **`config/`**: for configuration files for data preprocessing, model hyperparameters, paths, etc.
  - **`prediction/`**: Scripts for the forecaster model implemented using **PyTorch** library.
  - **`logger.py`**: This script contains the logger configuration using **logging** module.
  - **`train.py`**: This script is used to train the model. It loads the data, preprocesses it, trains the model, and saves the artifacts in the path `./model_inputs_outputs/model/artifacts/`.
  - **`predict.py`**: This script is used to run batch predictions using the trained model. It loads the artifacts and creates and saves the predictions in a file called `predictions.csv` in the path `./model_inputs_outputs/outputs/predictions/`.
  - **`utils.py`**: This script contains utility functions used by the other scripts.
- **`.gitignore`**: This file specifies the files and folders that should be ignored by Git.
- **`Dockerfile`**: This file is used to build the Docker image for the application.
- **`entry_point.sh`**: This file is used as the entry point for the Docker container. It is used to run the application. When the container is run using one of the commands `train`, `predict`, this script runs the corresponding script in the `src` folder to execute the task.
- **`LICENSE`**: This file contains the license for the project.
- **`requirements.txt`** for the main code in the `src` directory
- **`README.md`**: This file (this particular document) contains the documentation for the project, explaining how to set it up and use it.

## Usage

In this section we cover the following:

- How to prepare your data for training
- How to run the model implementation locally (without Docker)
- How to run the model implementation with Docker

### Preparing your data

- If you plan to run this model implementation on your own image classification dataset, you will need your training and testing data in a format similar to the one provided in **`/examples`**.

### To run locally (without Docker)

- Create your virtual environment and install dependencies listed in `requirements.txt` which is inside the `root` directory.
- Move the two example folders (`training` and `testing`) in the `examples` directory into the `./model_inputs_outputs/inputs` (or alternatively, place your custom dataset folders in the same location).
- Run the script `src/train.py` to train the classifier model. This will save the model artifacts, including the preprocessing pipeline and label encoder, in the path `./model_inputs_outputs/model/artifacts/`.
- Run the script `src/predict.py` to run batch predictions using the trained model. This script will load the artifacts and create and save the predictions in a file called `predictions.csv` in the path `./model_inputs_outputs/outputs/predictions/`.

### To run with Docker

1. Set up a bind mount on host machine: It needs to mirror the structure of the `model_inputs_outputs` directory. Place the data folders in the `model_inputs_outputs/inputs` directory.
2. Build the image. You can use the following command: <br/>
   `docker build -t model_img .` <br/>
   Here `model_img` is the name given to the container (you can choose any name).
3. Note the following before running the container for train, batch prediction:
   - The train, batch predictions tasks require a bind mount to be mounted to the path `/opt/model_inputs_outputs/` inside the container. You can use the `-v` flag to specify the bind mount.
   - When you run the train or batch prediction tasks, the container will exit by itself after the task is complete.
   - When you run training task on the container, the container will save the trained model artifacts in the specified path in the bind mount. This persists the artifacts even after the container is stopped or killed.
   - When you run the batch prediction task, the container will load the trained model artifacts from the same location in the bind mount. If the artifacts are not present, the container will exit with an error.
   - Container runs as user 1000. Provide appropriate read-write permissions to user 1000 for the bind mount. Please follow the principle of least privilege when setting permissions. The following permissions are required:
     - Read access to the `inputs` directory in the bind mount. Write or execute access is not required.
     - Read-write access to the `outputs` directory and `model` directories. Execute access is not required.
4. Run training:
   - To run training, run the container with the following command container: <br/>
     `docker run -v <path_to_mount_on_host>/model_inputs_outputs:/opt/model_inputs_outputs model_img train` <br/>
     where `model_img` is the name of the container. This will train the model and save the artifacts in the `model_inputs_outputs/model/artifacts` directory in the bind mount.
5. To run batch predictions, place the prediction data folder in the `model_inputs_outputs/inputs` directory in the bind mount. Then issue the command: <br/>
   `docker run -v <path_to_mount_on_host>/model_inputs_outputs:/opt/model_inputs_outputs model_img predict` <br/>
   This will load the artifacts and create and save the predictions in a file called `predictions.csv` in the path `model_inputs_outputs/outputs/predictions/` in the bind mount.

---

## Configuration Files
This project uses several configuration files to allow for easy customization and flexibility during training and prediction phases. Below, we describe the purpose and structure of these key configuration files.

**`model_config.json`**
You can specify your desired model through **`src/config/model_config.json`**. This file allows you to set various parameters, including the choice of the model architecture, seed value for reproducibility, and the field name for the predictions. Here is an example of the contents of model_config.json:
```json
{
  "model_name": "inceptionV3",
  "seed_value": 42,
  "prediction_field_name": "prediction"
}

```
Supported models include "resnet18", "resnet34", "resnet50", "resnet101", "resnet152", "inceptionV1", "inceptionV3", "mnasnet0_5", "mnasnet1_0", and "mnasnet1_3".

**`default_hyperparameters.json`**
Contains the default set of hyperparameters for training the model.

```json
{
  "lr": 0.1,
  "optimizer": "adam",
  "max_epochs": 10,
  "early_stopping": false,
  "early_stopping_patience": 3,
  "early_stopping_delta": 0.05,
  "lr_scheduler": "warmup_cosine_annealing",
  "lr_scheduler_kwargs": {
    "base_lr": 0.001,
    "warmup_epochs": 2,
    "num_epochs": 10
  }
}

```

- lr: Learning rate.
- optimizer: Optimization algorithm.
- max_epochs: Maximum number of epochs for training.
- early_stopping: Enables/disables early stopping.
- early_stopping_patience: Epochs to wait for improvement before stopping.
- early_stopping_delta: Minimum change to qualify as an improvement.
- lr_scheduler: Learning rate scheduling strategy.
- lr_scheduler_kwargs: Additional settings for the learning rate scheduler.

For detailed information, refer to the docstrings in the source code. 

**`preprocessing.json`**
Defines parameters for data preprocessing and augmentation.

```json
{
  "batch_size": 256,
  "num_workers": 0,
  "validation_size": 0.0
}
```
- batch_size: Number of samples processed in one iteration.
- num_workers: Number of subprocesses for data loading.
- validation_size: Portion of the dataset to use for validation.

By editing these files, users can customize the model's architecture, training parameters, and preprocessing steps to suit their specific needs, ensuring optimal performance for their image classification tasks.

---
## Requirements

Dependencies for the main model implementation in `src` are listed in the file `requirements.txt`.
You can install these packages by running the following command from the root of your project directory:

```python
pip install -r requirements.txt
```

## LICENSE

Please see the [LICENSE](LICENSE) file for more information.

## Contact Information

Repository created by Ready Tensor, Inc. Visit https://www.readytensor.ai/
