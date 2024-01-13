class FinetuningConfiguration:
    def __init__(self, model_config, trainer_logging_config, training_config, quantization_config, lora_config, tokenizer_config, mlflow_config):
        self.model_config = model_config
        self.trainer_logging_config = trainer_logging_config
        self.training_config = training_config
        self.quantization_config = quantization_config
        self.lora_config = lora_config
        self.tokenizer_config = tokenizer_config
        self.mlflow_config = mlflow_config
