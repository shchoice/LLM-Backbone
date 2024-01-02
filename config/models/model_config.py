class ModelConfig:
    def __init__(self, model_name, dataset, prompt_type):
        self.model_name = model_name
        self.dataset = dataset
        self.prompt_type = prompt_type

    def __str__(self):
        return f"ModelConfig(" \
               f"model_name={self.model_name}, " \
               f"dataset={self.dataset}, " \
               f"prompt_type={self.prompt_type})"
