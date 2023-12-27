from config.training_environment import TrainingEnvironment
from config.lora_config import LoraConfiguration
from config.model_config import ModelConfig
from config.quantization_config import QuantizationConfig
from config.tokenizer_config import TokenizerConfig
from config.train_config import TrainingConfig
from config.trainer_logging__config import TrainerLoggingConfig


class ConfigBuilder:
    def __init__(self):
        #  설정 객체를 바로 초기화하지 않는 이유: `ConfigBuilder`를 사용할 때 모든 설정을 한 번에 제공하지 않을 수도 있기 때문
        self.reset()

    def reset(self):
        self.model_config = None
        self.trainer_logging_config = None
        self.training_config = None
        self.quantization_config = None
        self.lora_config = None
        self.tokenizer_config = None

    def set_model_config(self, **kwargs):
        self.model_config = ModelConfig(**kwargs)
        return self

    def set_trainer_logging_config(self, **kwargs):
        self.trainer_logging_config = TrainerLoggingConfig(**kwargs)
        return self

    def set_training_config(self, **kwargs):
        self.training_config = TrainingConfig(**kwargs)
        return self

    def set_quantization_config(self, **kwargs):
        self.quantization_config = QuantizationConfig(**kwargs)
        return self

    def set_lora_config(self, **kwargs):
        self.lora_config = LoraConfiguration(**kwargs)
        return self

    def set_tokenizer_config(self, **kwargs):
        self.tokenizer_config = TokenizerConfig(**kwargs)
        return self

    def build(self):
        environment = TrainingEnvironment(
            self.model_config,
            self.trainer_logging_config,
            self.training_config,
            self.quantization_config,
            self.lora_config,
            self.tokenizer_config
        )
        self.reset()  # 다음 빌드를 위해 리셋

        return environment
