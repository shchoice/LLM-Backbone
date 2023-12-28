import os

import psutil

class OSEnvironmentUtils:
    @staticmethod
    def get_cpu_env():
        logical_cores = psutil.cpu_count(logical=True)
        physical_cores = psutil.cpu_count(logical=False)
        print(f"Number of logical cores: {logical_cores}")
        print(f"Number of physical cores: {physical_cores}")
        os.environ["OMP_NUM_THREADS"] = str(physical_cores)

    @staticmethod
    def set_mlflow_env(expt_name: str, model_name: str, datetime: str, mlflow_tracking_uri: str, mlflow_port: str):
        expt_name = expt_name + '_' + model_name + '_' + datetime
        os.environ['MLFLOW_EXPERIMENT_NAME'] = expt_name
        os.environ['MLFLOW_TRACKING_URI'] = mlflow_tracking_uri
        os.environ['MLFLOW_PORT'] = mlflow_port
