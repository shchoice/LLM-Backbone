# Product 클래스 정의 (최종 객체)
from config.lora_config import LoraConfiguration
from config.model_config import ModelConfig
from config.quantization_config import QuantizationConfig
from config.tokenizer_config import TokenizerConfig
from config.train_config import TrainingConfig
from config.trainer_logging__config import TrainerLoggingConfig


class TrainingEnvironment:
    def __init__(self,
                 model_config: ModelConfig,
                 trainer_logging_config: TrainerLoggingConfig,
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
