import argparse


def parse_list(arg_value):
    return arg_value.split(',')


def str2bool(v):
    if isinstance(v, bool):
       return v
    if v.lower() in ('yes', 'true', 't', 'y', '1'):
        return True
    elif v.lower() in ('no', 'false', 'f', 'n', '0'):
        return False
    else:
        raise argparse.ArgumentTypeError('Boolean value expected.')


class Arguments:
    @staticmethod
    def get_train_parse_arguments():
        parser = argparse.ArgumentParser(description='Finetuning model Question & Answering')

        # Select a Model and Dataset
        parser.add_argument('--model_name', type=str, required=True, choices=['koalpaca-12.8B', 'llama2-7B', 'llama2-13B'], help='Model name to use for finetuning')
        parser.add_argument('--dataset', type=str, required=True, default='KorQuAD-v1', help='Dataset to use for finetuning')

        # Select a Prompt
        parser.add_argument('--prompt_type', type=str, required=True, default='A', help='Prompt Type for Insturction Finetuning')

        # About a Log
        parser.add_argument('--expt_name', type=str, default='expt', help='Experiment name for output directory')
        parser.add_argument('--cache_dir', type=str, default='.cache', help='Cache directory for storing models and data')
        parser.add_argument('--output_dir', type=str, default='output', help='Output directory path')
        parser.add_argument('--logging_dir', type=str, default='logging', help='Logging directory path')
        parser.add_argument('--report_to', type=parse_list, default='mlflow,tensorboard', help='Report the results and logs')

        # For TrainerArguments
        parser.add_argument('--num_train_epochs', type=int, default=10, help='Total number of training epochs to perform')
        parser.add_argument('--train_batch_size', type=int, default=8, help='Training batch size')
        parser.add_argument('--eval_batch_size', type=int, default=4, help='Evaluation batch size')
        parser.add_argument('--evaluation_strategy', type=str, default='steps', choices=['no', 'steps', 'epoch'], help='Evaluation strategy to use')
        parser.add_argument('--eval_steps', type=int, default=200, help='Number of update steps')
        parser.add_argument('--save_steps', type=int, default=200, help='Number of updates steps before checkpoint saves')
        parser.add_argument('--logging_steps', type=int, default=200, help='Logging steps')
        parser.add_argument('--learning_rate', type=float, default=5e-4, help='Learning rate for the optimizer')
        parser.add_argument('--lr_scheduler_type', type=str, default='cosine',  choices=['constant', 'linear', 'cosine', 'exponential', 'polynomial'], help='Type of learning rate scheduler to use during training')
        parser.add_argument('--optim', type=str, default='paged_adamw_8bit', choices=['adamw_torch', 'paged_adamw_8bit'], help='Optimizer type')
        parser.add_argument('--warmup_ratio', type=float, default=0.1, help='Warmup ratio for learning rate scheduler')
        parser.add_argument('--weight_decay', type=float, default=0.01, help='Weight decay for the optimizer')
        parser.add_argument('--gradient_accumulation_steps', type=int, default=4, help='Gradient accumulation steps')
        parser.add_argument('--load_best_model_at_end', type=str2bool, default=True, help='Load best model at end of training')
        parser.add_argument('--fp16', type=str2bool, default=True, help='Whether to use fp16 (mixed) precision instead of 32-bit')
        parser.add_argument('--ddp_find_unused_parameters', type=str2bool, default=False, help='When using distributed training, the value of the flag `find_unused_parameters` passed to DistributedDataParallel')
        parser.add_argument('--early_stopping_patience', type=int, default=10, help='Early stopping patience')

        # For Transformers Quantization Config
        parser.add_argument('--load_in_4bit', type=str2bool, default=True, help='Enable 4-bit quantization by replacing the Linear layers with FP4/NF4 layers from bitsandbytes.')
        parser.add_argument('--bnb_4bit_quant_type', type=str, default='nf4', choices=['fp4', 'nf4'], help='quantization data type in the bnb.nn.Linear4Bit layers. Options are FP4 and NF4 data types which are specified by `fp4` or `nf4`.')
        parser.add_argument('--bnb_4bit_compute_dtype', type=str, default='torch.bfloat16', choices=['torch.bfloat16', 'torch.float32'], help='Computational type which might be different than the input time. For example, inputs might be fp32, but computation can be set to bf16 for speedups')
        parser.add_argument('--bnb_4bit_use_double_quant', type=str2bool, default=True, help='Used for nested quantization where the quantization constants from the first quantization are quantized again')

        # For Lora Config
        parser.add_argument('--r', type=int, default=8, help='Lora attention dimension')
        parser.add_argument('--lora_alpha', type=int, default=32, help='The alpha parameter for Lora scaling')
        parser.add_argument('--lora_dropout', type=float, default=0.05, help='The dropout probability for Lora layers')
        parser.add_argument('--fan_in_fan_out', type=str2bool, default=False, help='Set this to True if the layer to replace stores weight like (fan_in, fan_out)')
        parser.add_argument('--bias', type=str, default='none', help='Bias type for Lora. Can be \'none\', \'all\' or \'lora_only\'. If \'all\' or \'lora_only\', the corresponding biases will be updated during training. Be aware that this means that, even when disabling the adapters, the model will not produce the same output as the base model would have without adaptation')
        parser.add_argument('--target_modules', type=parse_list, default='query_key_value', help='The names of the modules to apply Lora to')
        parser.add_argument('--inference_mode', type=str2bool, default=False, help='Flag to set the model in inference mode. When True, the model will be used for making predictions rather than training.')
        parser.add_argument('--task_type', type=str, default='CAUSAL_LM', choices=['CAUSAL_LM', 'CLASSIFICATION', 'QUESTION_ANSWERING', 'NAMED_ENTITY_RECOGNITION', 'SENTIMENT_ANALYSIS'], help='Type of task to perform. For example, "CAUSAL_LM" for causal language modeling. Other tasks might include "CLASSIFICATION", "QUESTION_ANSWERING", "NAMED_ENTITY_RECOGNITION", "SENTIMENT_ANALYSIS", etc.')

        # For Tokenizer
        parser.add_argument('--max_length', type=int, default=2048, help='Maximum sequence length for the tokenizer')
        parser.add_argument('--truncation', type=str2bool, default=True, help='Enable or disable truncation')
        parser.add_argument('--return_overflowing_tokens', type=str2bool, default=True, help='Return overflowing token information')
        parser.add_argument('--return_length', type=str2bool, default=True, help='Return length of the encoded inputs')
        parser.add_argument('--padding', type=str2bool, default=True, help='Enable padding to the maximum sequence length')

        # For MLFLOW
        parser.add_argument('--mlflow_tracking_uri', type=str, default='http://localhost:5000', help='URI of MLFlow installed')

        return parser.parse_args()

    @staticmethod
    def get_test_parse_arguments():
        pass
