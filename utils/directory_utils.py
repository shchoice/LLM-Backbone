import os
from datetime import datetime

from utils.experiment_datetime_utils import ExperimentDatetimeUtils


class DirectoryUtils:
    @staticmethod
    def get_base_path():
        current_file = os.path.abspath(__file__)
        base_dir = os.path.dirname(os.path.dirname(current_file))

        return base_dir

    @staticmethod
    def get_cache_dir(cache_dir):
        base_dir = DirectoryUtils.get_base_path()
        cache_dir = os.path.join(base_dir, cache_dir)

        return cache_dir

    @staticmethod
    def get_output_dir(model_name, expt_name, output_dir):
        # Expect: OUTPUT_DIR = f'./output/{MODEL_NAME}/{EXPT_NAME}_{datetime.now().strftime("%y%m%d_%H%M")}'
        base_dir = DirectoryUtils.get_base_path()
        date = ExperimentDatetimeUtils.get_experiment_datetime()
        output_dir = os.path.join(
            base_dir,
            output_dir,
            model_name,
            expt_name + '_' + date,
        )

        return output_dir

    @staticmethod
    def get_logging_dir(model_name, expt_name, output_dir, logging_dir):
        # Expect: LOGGING_DIR = f'./output/{MODEL_NAME}/{EXPT_NAME}_{datetime.now().strftime("%y%m%d_%H%M")/logging}'
        base_dir = DirectoryUtils.get_base_path()
        date = ExperimentDatetimeUtils.get_experiment_datetime()
        logging_dir = os.path.join(
            base_dir,
            output_dir,
            model_name,
            expt_name + '_' + date,
            logging_dir
        )

        return logging_dir
