import numpy as np
import pandas as pd

from config import paths
from logger import get_logger, log_error
from prediction.predictor_model import predict_with_model, load_predictor_model
from utils import (
    save_dataframe_as_csv,
    ResourceTracker,
    create_predictions_dataframe,
)
from data_loader.data_loader import load_data_loader_factory

logger = get_logger(task_name="predict")


def run_batch_predictions(
    test_dir_path: str = paths.TEST_DIR,
    predictor_dir_path: str = paths.PREDICTOR_DIR_PATH,
    predictions_file_path: str = paths.PREDICTIONS_FILE_PATH,
    data_loader_file_path: str = paths.SAVED_DATA_LOADER_FILE_PATH,
) -> None:
    """
    Run batch predictions on test data, save the predicted probabilities to a CSV file.

    Args:
        test_dir_path (str): Directory path for the test data.
        predictor_dir_path (str): Path to the directory of saved model.
        predictions_file_path (str): Path where the predictions file will be saved.
        data_loader_file_path (str): Path to the saved data loader file.
    """

    try:
        with ResourceTracker(logger, monitoring_interval=5):
            logger.info("Making batch predictions...")

            logger.info("Loading test data...")
            data_loader = load_data_loader_factory(
                data_loader_file_path=data_loader_file_path
            )
            test_data, image_names = data_loader.create_test_data_loader(
                data_dir_path=test_dir_path
            )

            logger.info("Loading predictor model...")
            predictor_model = load_predictor_model(predictor_dir_path)

            logger.info("Making predictions...")
            predicted_labels, predicted_probabilities = predict_with_model(
                predictor_model, test_data
            )

            logger.info("Creating final predictions dataframe...")
            predictions_df = create_predictions_dataframe(
                ids=image_names,
                probs=predicted_probabilities,
                predictions=predicted_labels,
                class_to_idx=data_loader.class_to_idx,
            )

        logger.info("Saving predictions dataframe...")
        save_dataframe_as_csv(dataframe=predictions_df, file_path=predictions_file_path)

    except Exception as exc:
        err_msg = "Error occurred during prediction."
        # Log the error
        logger.error(f"{err_msg} Error: {str(exc)}")
        # Log the error to the separate logging file
        log_error(message=err_msg, error=exc, error_fpath=paths.PREDICT_ERROR_FILE_PATH)
        # re-raise the error
        raise Exception(f"{err_msg} Error: {str(exc)}") from exc


if __name__ == "__main__":
    run_batch_predictions()
