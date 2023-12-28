from config.models.lora_config import LoraConfiguration
from config.models.model_config import ModelConfig
from config.models.quantization_config import QuantizationConfig
from config.training.tokenizer_config import TokenizerConfig
from config.training.training_config import TrainingConfig
from config.training.training_logging_config import TrainingLoggingConfig


class TrainingEnvironment:
    def __init__(self,
                 model_config: ModelConfig,
                 trainer_logging_config: TrainingLoggingConfig,
                 training_config: TrainingConfig,
                 quantization_config: QuantizationConfig,
                 lora_config: LoraConfiguration,
                 tokenizer_config: TokenizerConfig):
        self.model_config = model_config
        self.trainer_logging_config = trainer_logging_config
        self.training_config = training_config
        self.quantization_config = quantization_config
        self.lora_config = lora_config
        self.tokenizer_config = tokenizer_config
