class ModelConfig:
    def __init__(self, model_name, dataset_path, prompt_type):
        self.model_name = model_name
        self.dataset_path = dataset_path
        self.prompt_type = prompt_type

    def __str__(self):
        return f"ModelConfig(" \
               f"model_name={self.model_name}, " \
               f"dataset={self.dataset_path}, " \
               f"prompt_type={self.prompt_type})"
