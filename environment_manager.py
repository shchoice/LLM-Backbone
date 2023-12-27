import os

import psutil


class EnvironmentManager:
    @staticmethod
    def set_environment_variables():
        logical_cores = psutil.cpu_count(logical=True)
        physical_cores = psutil.cpu_count(logical=False)
        print(f"Number of logical cores: {logical_cores}")
        print(f"Number of physical cores: {physical_cores}")
        os.environ["OMP_NUM_THREADS"] = str(physical_cores)
