import os
from config import paths
from logger import get_logger, log_error
from prediction.predictor_model import (
    save_predictor_model,
    train_predictor_model,
)
from torch_utils.data_loader import CustomDataLoader

from utils import (
    read_json_as_dict,
    set_seeds,
    contains_subdirectories,
    ResourceTracker,
)

logger = get_logger(task_name="train")

validation_exists = os.path.isdir(paths.VALIDATION_DIR) and contains_subdirectories(
    paths.VALIDATION_DIR
)


def run_training(
    model_config_file_path: str = paths.MODEL_CONFIG_FILE_PATH,
    train_dir_path: str = paths.TRAIN_DIR,
    valid_dir_path: str = paths.VALIDATION_DIR,
    preprocessing_config_file_path: str = paths.PREPROCESSING_CONFIG_FILE_PATH,
    predictor_dir_path: str = paths.PREDICTOR_DIR_PATH,
    default_hyperparameters_file_path: str = paths.DEFAULT_HYPERPARAMETERS_FILE_PATH,
    data_loader_save_path: str = paths.SAVED_DATA_LOADER_FILE_PATH,
) -> None:
    """
    Run the training process and saves model artifacts

    Args:
        input_schema_dir (str, optional): The directory path of the input schema.
        saved_schema_dir_path (str, optional): The path where to save the schema.
        model_config_file_path (str, optional): The path of the model
            configuration file.
        train_dir (str, optional): The directory path of the train data.
        predictor_dir_path (str, optional): Dir path where to save the
            predictor model.
        default_hyperparameters_file_path (str, optional): The path of the default
            hyperparameters file.
    Returns:
        None
    """

    try:
        with ResourceTracker(logger=logger, monitoring_interval=5):
            logger.info("Starting training...")

            # load model config
            logger.info("Loading model config...")
            model_config = read_json_as_dict(model_config_file_path)

            logger.info("Loading preprocessing config...")
            preprocessing_config = read_json_as_dict(preprocessing_config_file_path)

            # set seeds
            logger.info("Setting seeds...")
            set_seeds(seed_value=model_config["seed_value"])

            # load train data and validation data if available
            logger.info("Loading train data...")
            data_loader = CustomDataLoader(**preprocessing_config)
            if validation_exists:
                train_data = data_loader.create_data_loader(
                    data_dir_path=train_dir_path,
                    shuffle=True,
                )
                valid_data = data_loader.create_data_loader(
                    data_dir_path=valid_dir_path,
                    shuffle=False,
                )

            elif (
                not validation_exists
                and "validation_size" in preprocessing_config
                and preprocessing_config["validation_size"] > 0
            ):
                train_data, valid_data = data_loader.create_data_loader(
                    data_dir_path=train_dir_path,
                    shuffle=True,
                    create_validation=True,
                    val_size=preprocessing_config["validation_size"],
                )
            else:
                train_data = data_loader.create_data_loader(
                    data_dir_path=train_dir_path,
                    shuffle=True,
                )
                valid_data = None

            # use default hyperparameters to train model
            logger.info("Loading hyperparameters...")
            default_hyperparameters = read_json_as_dict(
                default_hyperparameters_file_path
            )

            # # use default hyperparameters to train model
            logger.info("Training model...")
            model = train_predictor_model(
                train_data=train_data,
                valid_data=valid_data,
                num_classes=data_loader.num_classes,
                hyperparameters=default_hyperparameters,
            )

        # save data loader
        logger.info("Saving data loader...")
        data_loader.save(data_loader_save_path)

        # save predictor model
        logger.info("Saving model...")
        save_predictor_model(model, predictor_dir_path)

    except Exception as exc:
        err_msg = "Error occurred during training."
        # Log the error
        logger.error(f"{err_msg} Error: {str(exc)}")
        # Log the error to the separate logging file
        log_error(message=err_msg, error=exc, error_fpath=paths.TRAIN_ERROR_FILE_PATH)
        # re-raise the error
        raise Exception(f"{err_msg} Error: {str(exc)}") from exc


if __name__ == "__main__":
    run_training()
