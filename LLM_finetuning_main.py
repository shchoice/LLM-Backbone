import sys

from transformers import Trainer


from config.entity.training.finetuning_configuration import FinetuningConfiguration
from config.training.training_arguments_manager import TrainingArgumentsManager
from dataset.dataset_manager import DatasetManager
from LLM_finetuning_main_arguments import get_debug_arguments, get_experiment_arguments
from models.model_factory import ModelFactory
from tokenizer.tokenizer_manager import TokenizerManager
from utils.experiment_datetime_utils import ExperimentDatetimeUtils
from utils.os_environment_utils import OSEnvironmentUtils


class FinetuningService:
    def __init__(self, config: FinetuningConfiguration):
        self.config = config

        self.model = None
        self.model_manager = ModelFactory.create_model(config=config)

        self.tokenizer = None
        self.tokenizer_manager = TokenizerManager(config=config)

        self.dataset_module = None
        self.data_loader = DatasetManager(config)
        total_samples = len(self.data_loader.dataset)
        training_config.set_warmup_steps(total_samples)

        self.training_arguments_manager = TrainingArgumentsManager(config)

        self.early_stopping_patience = training_config.early_stopping_patience
        datetime = ExperimentDatetimeUtils.get_experiment_datetime()
        OSEnvironmentUtils.set_mlflow_env(
            expt_name=trainer_logging_config.expt_name,
            model_name=model_config.model_name,
            datetime=datetime,
            mlflow_tracking_uri=mlflow_config.mlflow_tracking_uri,
        )

    def setup(self):
        self.model = self.model_manager.load_model()
        self.tokenizer = self.tokenizer_manager.load_tokenizer()
        self.dataset_module = self.data_loader.make_supervised_dataset()

    def train(self):
        trainer = Trainer(
            model=self.model,
            tokenizer=self.tokenizer,
            args=self.training_arguments_manager.get_training_arguments(),
            train_dataset=self.dataset_module['train_dataset'],
            eval_dataset=self.dataset_module['eval_dataset'],
            data_collator=self.dataset_module['data_collator'],
        )

        self.model.config.use_cache = False
        trainer.train()


if __name__ == '__main__':
    OSEnvironmentUtils.get_cpu_env()

    if len(sys.argv) == 1:
        get_debug_arguments()

    model_config, trainer_logging_config, training_config, quantization_config, lora_config, tokenizer_config, \
        mlflow_config = get_experiment_arguments()

    finetuning_configuration = FinetuningConfiguration(
        model_config=model_config,
        trainer_logging_config=trainer_logging_config,
        training_config=training_config,
        quantization_config=quantization_config,
        lora_config=lora_config,
        tokenizer_config=tokenizer_config,
        mlflow_config=mlflow_config,
    )

    finetuning_service = FinetuningService(config=finetuning_configuration)
    finetuning_service.setup()
    finetuning_service.train()
