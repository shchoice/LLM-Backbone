import sys

from transformers import (
    DataCollatorForLanguageModeling,
    Trainer,
    EarlyStoppingCallback,
)

from config.config_builder import ConfigBuilder
from config.lora_config import LoraConfiguration
from config.model_config import ModelConfig
from config.quantization_config import QuantizationConfig
from config.tokenizer_config import TokenizerConfig
from config.train_config import TrainingConfig
from config.trainer_logging__config import TrainerLoggingConfig
from config.arguments import Arguments
from config.training_config_manager import TrainerConfigManager
from data_loader import DataLoader
from environment_manager import EnvironmentManager
from model_loader import ModelLoader
import mlflow
import os

os.environ['MLFLOW_EXPERIMENT_NAME']="test"
os.environ['MLFLOW_FLATTEN_PARAMS']='1'

class QuestionAndAnsweringService:
    def __init__(self, model_config: ModelConfig,
                 trainer_logging_config: TrainerLoggingConfig,
                 training_config: TrainingConfig,
                 quantization_config: QuantizationConfig,
                 lora_config: LoraConfiguration,
                 tokenizer_config: TokenizerConfig):
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
            prompt_name=model_config.prompt_name,
            cache_dir=trainer_logging_config.cache_dir
        )
        total_samples = len(self.data_loader.dataset)
        training_config.set_warmup_steps(total_samples)

        self.training_arguments_manager = TrainerConfigManager(
            model_name= model_config.model_name,
            training_logging_config=trainer_logging_config,
            training_config=training_config
        )

        self.model = None
        self.tokenizer = None
        self.dataset = None
        self.early_stopping_patience = training_config.early_stopping_patience

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

        # self.model.save_pretrained()


def setup_training_environment(args):
    config_builder = ConfigBuilder()
    config_builder \
        .set_model_config(
            model_name=args.model_name,
            dataset=args.dataset,
            prompt_name=args.prompt_name
        ) \
        .set_trainer_logging_config(
            expt_name=args.expt_name,
            cache_dir=args.cache_dir,
            output_dir=args.output_dir,
            logging_dir=args.logging_dir,
            report_to=args.report_to
        ) \
        .set_training_config(
            num_train_epochs=args.num_train_epochs,
            train_batch_size=args.train_batch_size,
            eval_batch_size=args.eval_batch_size,
            evaluation_strategy=args.evaluation_strategy,
            eval_steps=args.eval_steps,
            save_steps=args.save_steps,
            logging_steps=args.logging_steps,
            learning_rate=args.learning_rate,
            lr_scheduler_type=args.lr_scheduler_type,
            optim=args.optim,
            warmup_ratio=args.warmup_ratio,
            weight_decay=args.weight_decay,
            gradient_accumulation_steps=args.gradient_accumulation_steps,
            load_best_model_at_end=args.load_best_model_at_end,
            fp16=args.fp16,
            ddp_find_unused_parameters=args.ddp_find_unused_parameters,
            early_stopping_patience=args.early_stopping_patience
        ) \
        .set_quantization_config(
            load_in_4bit=args.load_in_4bit,
            bnb_4bit_quant_type=args.bnb_4bit_quant_type,
            bnb_4bit_compute_dtype=args.bnb_4bit_compute_dtype,
            bnb_4bit_use_double_quant=args.bnb_4bit_use_double_quant
        ) \
        .set_lora_config(
            r=args.r,
            lora_alpha=args.lora_alpha,
            lora_dropout=args.lora_dropout,
            fan_in_fan_out=args.fan_in_fan_out,
            bias=args.bias,
            target_modules=args.target_modules,
            inference_mode=args.inference_mode,
            task_type=args.task_type
        ) \
        .set_tokenizer_config(
            max_length=args.max_length,
            truncation=args.truncation,
            return_overflowing_tokens=args.return_overflowing_tokens,
            return_length=args.return_length,
            padding=args.padding
        )
    return config_builder.build()

def print_training_environment(training_environment):
    print(training_environment.model_config)
    print(training_environment.trainer_logging_config)
    print(training_environment.training_config)
    print(training_environment.quantization_config)
    print(training_environment.lora_config)
    print(training_environment.tokenizer_config)


if __name__ == '__main__':
    EnvironmentManager.set_environment_variables()

    if len(sys.argv) == 1:  # 디버깅 환경에서 사용할 경우(파라미터가 제공되지 않은 경우)
        test_args = ['--model_name', 'koalpaca-12.8B',
                     '--dataset', 'KorQuAD-v1',
                     '--prompt_name', 'A'
                     ]
        sys.argv.extend(test_args)  # 테스트 인자 추가

    args = Arguments.get_train_parse_arguments()
    training_environment = setup_training_environment(args=args)
    print_training_environment(training_environment=training_environment)

    qa_service = QuestionAndAnsweringService(
        model_config=training_environment.model_config,
        trainer_logging_config=training_environment.trainer_logging_config,
        training_config=training_environment.training_config,
        quantization_config=training_environment.quantization_config,
        lora_config=training_environment.lora_config,
        tokenizer_config=training_environment.tokenizer_config
    )
    qa_service.setup()
    qa_service.train()
