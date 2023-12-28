from transformers import TrainingArguments

from config.training.training_config import TrainingConfig
from config.training.training_logging_config import TrainingLoggingConfig
from utils.directory_utils import DirectoryUtils


class TrainingConfigManager:
    def __init__(self, model_name, training_logging_config: TrainingLoggingConfig, training_config: TrainingConfig):
        self.model_name = model_name
        self.training_logging_config = training_logging_config
        self.training_config = training_config

        self.output_dir = DirectoryUtils.get_output_dir(
            model_name=self.model_name,
            expt_name=self.training_logging_config.expt_name,
            output_dir=self.training_logging_config.output_dir
        )
        self.logging_dir = DirectoryUtils.get_logging_dir(
            model_name=self.model_name,
            expt_name=self.training_logging_config.expt_name,
            logging_dir=self.training_logging_config.logging_dir
        )

    def get_training_arguments(self):
        args = TrainingArguments(
            output_dir=self.output_dir,
            logging_dir=self.logging_dir,
            report_to=self.training_logging_config.report_to,

            num_train_epochs=self.training_config.num_train_epochs,
            per_device_train_batch_size=self.training_config.train_batch_size,
            per_device_eval_batch_size=self.training_config.eval_batch_size,

            evaluation_strategy=self.training_config.evaluation_strategy,
            eval_steps=self.training_config.eval_steps,
            save_steps=self.training_config.save_steps,
            logging_steps=self.training_config.logging_steps,

            learning_rate=self.training_config.learning_rate,
            lr_scheduler_type=self.training_config.lr_scheduler_type,
            optim=self.training_config.optim,

            warmup_ratio=self.training_config.warmup_ratio,
            warmup_steps=self.training_config.warmup_steps,

            weight_decay=self.training_config.weight_decay,

            gradient_accumulation_steps=self.training_config.gradient_accumulation_steps,
            load_best_model_at_end=self.training_config.load_best_model_at_end,
            fp16=self.training_config.fp16,
            ddp_find_unused_parameters=self.training_config.ddp_find_unused_parameters,
        )

        return args
