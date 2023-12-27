from utils.directory_utils import DirectoryUtils


class TrainerLoggingConfig:
    def __init__(self, expt_name, cache_dir, output_dir, logging_dir, report_to):
        self.expt_name = expt_name
        self.cache_dir = DirectoryUtils.get_cache_dir(cache_dir=cache_dir)
        self.output_dir = output_dir
        self.logging_dir = logging_dir
        self.report_to = report_to

    def __str__(self):
        return f"TrainerLoggingConfig(" \
               f"expt_name={self.expt_name}, " \
               f"cache_dir={self.cache_dir}, " \
               f"output_dir={self.output_dir}, " \
               f"logging_dir={self.logging_dir}, " \
               f"report_to={self.report_to})"
