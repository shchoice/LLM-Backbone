import sys

from transformers import DataCollatorForLanguageModeling, Trainer, EarlyStoppingCallback

from config.ml_ops.mlflow_config import MLFlowConfig
from config.models.lora_config import LoraConfiguration
from config.models.model_config import ModelConfig
from config.models.quantization_config import QuantizationConfig
from config.training.tokenizer_config import TokenizerConfig
from config.training.training_config import TrainingConfig
from config.training.training_config_manager import TrainingConfigManager
from config.training.training_logging_config import TrainingLoggingConfig
from data.data_loader import DataLoader
from LLM_finetuning_main_arguments import get_debug_arguments, get_experiment_arguments
from models.model_loader import ModelLoader
from utils.experiment_datetime_utils import ExperimentDatetimeUtils
from utils.os_environment_utils import OSEnvironmentUtils


class FinetuningService:
    def __init__(self, model_config: ModelConfig,
                 trainer_logging_config: TrainingLoggingConfig,
                 training_config: TrainingConfig,
                 quantization_config: QuantizationConfig,
                 lora_config: LoraConfiguration,
                 tokenizer_config: TokenizerConfig,
                 mlflow_config: MLFlowConfig):
        self.model_loader = ModelLoader(
            model_config=model_config,
            trainer_logging_config=trainer_logging_config,
            training_config=training_config,
            quantization_config=quantization_config,
            lora_config=lora_config,
            tokenizer_config=tokenizer_config
        )
        self.data_loader = DataLoader(
            dataset=model_config.dataset,
            prompt_type=model_config.prompt_type,
            cache_dir=trainer_logging_config.cache_dir
        )
        total_samples = len(self.data_loader.dataset)
        training_config.set_warmup_steps(total_samples)

        self.training_arguments_manager = TrainingConfigManager(
            model_name=model_config.model_name,
            training_logging_config=trainer_logging_config,
            training_config=training_config
        )

        self.model = None
        self.tokenizer = None
        self.dataset = None
        self.early_stopping_patience = training_config.early_stopping_patience
        datetime = ExperimentDatetimeUtils.get_experiment_datetime()
        OSEnvironmentUtils.set_mlflow_env(
            expt_name=trainer_logging_config.expt_name,
            model_name=model_config.model_name,
            datetime=datetime,
            mlflow_tracking_uri=mlflow_config.mlflow_tracking_uri,
        )

    def setup(self):
        self.model = self.model_loader.load_model()
        self.tokenizer = self.model_loader.load_tokenizer()
        self.dataset = self.data_loader.load_and_format_dataset()

    def train(self):
        inputs = self.dataset.map(self.model_loader.convert_to_inputs,
                                  batched=True,
                                  remove_columns=self.dataset['train'].column_names)

        data_collator = DataCollatorForLanguageModeling(self.tokenizer, mlm=False)
        training_batch = data_collator([inputs['train'][i] for i in range(3)])
        for key in training_batch:
            print(f"{key} : {training_batch[key].shape}")

        trainer = Trainer(
            model=self.model,
            tokenizer=self.tokenizer,
            args=self.training_arguments_manager.get_training_arguments(),
            data_collator=data_collator,
            train_dataset=inputs['train'],
            eval_dataset=inputs['validation'],
            callbacks=[EarlyStoppingCallback(early_stopping_patience=self.early_stopping_patience)]
        )

        self.model.config.use_cache = False
        trainer.train()


if __name__ == '__main__':
    OSEnvironmentUtils.get_cpu_env()

    if len(sys.argv) == 1:
        get_debug_arguments()

    # TODO
    ## Task를 적어야함 지금은 범용적인 한국어 Finetuning을 위해 작업해야하지만
    ## QnA, Summary 등의 DownStream Task에 대해서 작업을 사용하기 위함
    model_config, trainer_logging_config, training_config, quantization_config, lora_config, tokenizer_config, \
        mlflow_config = get_experiment_arguments()

    finetuning_service = FinetuningService(
        model_config=model_config,
        trainer_logging_config=trainer_logging_config,
        training_config=training_config,
        quantization_config=quantization_config,
        lora_config=lora_config,
        tokenizer_config=tokenizer_config,
        mlflow_config=mlflow_config,
    )

    finetuning_service.setup()
    finetuning_service.train()
