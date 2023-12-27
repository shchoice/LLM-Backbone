class ModelConfig:
    def __init__(self, model_name, dataset, prompt_name):
        self.model_name = model_name
        self.dataset = dataset
        self.prompt_name = prompt_name

    def __str__(self):
        return f"ModelConfig(" \
               f"model_name={self.model_name}, " \
               f"dataset={self.dataset}, " \
               f"prompt_name={self.prompt_name})"
