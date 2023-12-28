from datetime import datetime


class ExperimentDatetimeUtils:
    _experiment_time = datetime.now().strftime('%y%m%d_%H%M')

    @classmethod
    def get_experiment_datetime(cls):
        return cls._experiment_time
