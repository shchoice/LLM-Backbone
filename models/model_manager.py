import os

import torch
from peft import (
    prepare_model_for_kbit_training,
    LoraConfig,
    get_peft_model
)
from transformers import (
    AutoModelForCausalLM,
    BitsAndBytesConfig
)

from config import constants
from config.entity.models.lora_config import LoraConfiguration
from config.entity.models.model_config import ModelConfig
from config.entity.models import QuantizationConfig
from config.entity.training.finetuning_configuration import FinetuningConfiguration
from config.entity.training.training_config import TrainingConfig
from config.entity.training.training_logging_config import TrainingLoggingConfig


class ModelManager:
    def __init__(self, config: FinetuningConfiguration):
        model_config: ModelConfig = config
        trainer_logging_config: TrainingLoggingConfig = config
        training_config: TrainingConfig  = config
        quantization_config: QuantizationConfig = config
        lora_config: LoraConfiguration = config

        self.model = self.get_model(model_config.model_name)
        self.cache_dir = trainer_logging_config.cache_dir

        self.quantization_config = quantization_config
        self.lora_config = lora_config

        self.ddp = False
        self.gradient_accumulation_steps = training_config.gradient_accumulation_steps
        self.train_batch_size = training_config.train_batch_size

    def get_model(self, model_name):
        if model_name == constants.MODEL_NAME_KOALPACA_POLYGLOT_12_8B:
            return constants.HF_MODEL_CARD_NAME_KOALPACA_POLYGLOT_12_8B
        elif model_name == constants.MODEL_NAME_LLAMA_2_7B:
            return constants.HF_MODEL_CARD_NAME_LLAMA_2_7B

    def load_model(self):
        bnb_config = BitsAndBytesConfig(
            load_in_4bit=self.quantization_config.load_in_4bit,
            bnb_4bit_use_double_quant=self.quantization_config.bnb_4bit_use_double_quant,
            bnb_4bit_quant_type=self.quantization_config.bnb_4bit_quant_type,
            bnb_4bit_compute_dtype=self.get_torch_dtype(self.quantization_config.bnb_4bit_compute_dtype)
        )

        model = AutoModelForCausalLM.from_pretrained(
            self.model,
            quantization_config=bnb_config,
            device_map=self.get_device_map(),
            cache_dir=self.cache_dir
        )

        self.print_trainable_parameters(model)
        model.gradient_checkpointing_enable()
        model = prepare_model_for_kbit_training(model)

        config = LoraConfig(
            r=self.lora_config.r,
            lora_alpha=self.lora_config.lora_alpha,
            target_modules=self.lora_config.target_modules,
            fan_in_fan_out=self.lora_config.fan_in_fan_out,
            lora_dropout=self.lora_config.lora_dropout,
            inference_mode=self.lora_config.inference_mode,
            bias=self.lora_config.bias,
            task_type=self.lora_config.task_type
        )
        model = get_peft_model(model, config)
        self.print_trainable_parameters(model)

        if not self.ddp and torch.cuda.device_count() > 1:
            model.is_parallelizable = True
            model.model_parallel = True
            print("not ddp - trying its own DataParallelism")

        return model

    @staticmethod
    def print_trainable_parameters(model):
        trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
        all_params = sum(p.numel() for p in model.parameters())
        print(
            f"trainable params: {trainable_params} || all params: {all_params} || trainable%: {100 * trainable_params / all_params}"
        )

    @staticmethod
    def get_torch_dtype(dtype_str):
        if dtype_str == 'torch.bfloat16':
            return torch.bfloat16
        elif dtype_str == 'torch.float32':
            return torch.float32
        else:
            raise ValueError(f"Unsupported data type: {dtype_str}")

    def get_device_map(self):
        print(f"num_gpus: {torch.cuda.device_count()}")
        world_size = int(os.environ.get("WORLD_SIZE", torch.cuda.device_count()))
        print(f"world_size: {world_size}")
        self.ddp = world_size != 1
        if self.ddp:
            device_map = {"": int(os.environ.get("LOCAL_RANK") or 0)}
            self.gradient_accumulation_steps = self.train_batch_size // world_size
            if self.gradient_accumulation_steps == 0:
                self.gradient_accumulation_steps = 1
            print(f"ddp is on - gradient_accumulation_steps: {self.gradient_accumulation_steps}")
        else:
            device_map = "auto"
            print("ddp is off")

        return device_map
