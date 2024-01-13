from config.entity.ml_ops.mlflow_config import MLFlowConfig
from config.entity.models.lora_config import LoraConfiguration
from config.entity.models.model_config import ModelConfig
from config.entity.models.quantization_config import QuantizationConfig
from config.entity.tokenizer.tokenizer_config import TokenizerConfig
from config.entity.training.training_config import TrainingConfig
from config.entity.training.training_logging_config import TrainingLoggingConfig


class TrainingEnvironment:
    def __init__(self,
                 model_config: ModelConfig,
                 trainer_logging_config: TrainingLoggingConfig,
                 training_config: TrainingConfig,
                 quantization_config: QuantizationConfig,
                 lora_config: LoraConfiguration,
                 tokenizer_config: TokenizerConfig,
                 mlflow_config: MLFlowConfig):
        self.model_config = model_config
        self.trainer_logging_config = trainer_logging_config
        self.training_config = training_config
        self.quantization_config = quantization_config
        self.lora_config = lora_config
        self.tokenizer_config = tokenizer_config
        self.mlflow_config = mlflow_config
