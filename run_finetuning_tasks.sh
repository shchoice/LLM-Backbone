# Set the arguments
MODEL_NAME="koalpaca-12.8B"                         # Model name
DATASET="KorQuAD-v1"                                # Dataset

PROMPT_TYPE="A"                                     # Prompt Type

EXPT_NAME="expt"                                    # Experiment name
CACHE_DIR=".cache"                                  # Cache directory
OUTPUT_DIR="output"                                 # Output directory
LOGGING_DIR="logging"                               # Logging directory
REPORT_TO="mlflow,tensorboard"                      # Report the results and logs

NUM_TRAIN_EPOCHS=10                                 # Training epochs
TRAIN_BATCH_SIZE=8                                  # Training batch size
EVAL_BATCH_SIZE=4                                   # Evaluation batch size
EVALUATION_STRATEGY="steps"                         # Evaluation strategy
EVAL_STEPS=200                                      # Evaluation steps
SAVE_STEPS=200                                      # Save steps
LOGGING_STEPS=200                                   # Logging steps
LEARNING_RATE=5e-4                                  # Learning rate
LR_SCHEDULER_TYPE="cosine"                          # LR scheduler type
OPTIM="paged_adamw_8bit"                            # Optimizer type
WARMUP_RATIO=0.1                                    # Warmup ratio
WEIGHT_DECAY=0.05                                   # Weight decay
GRADIENT_ACCUMULATION_STEPS=4                       # Gradient accumulation steps
LOAD_BEST_MODEL_AT_END=true                         # Load best model at end
FP16=true                                           # Use fp16
DDP_FIND_UNUSED_PARAMETERS=false                    # DDP find unused parameters
EARLY_STOPPING_PATIENCE=10                          # Early stopping patience

LOAD_IN_4BIT=true                                   # Enable 4-bit quantization
BNB_4BIT_QUANT_TYPE="nf4"                           # BNB 4-bit quantization type
BNB_4BIT_COMPUTE_DTYPE="torch.bfloat16"             # BNB 4-bit compute dtype
BNB_4BIT_USE_DOUBLE_QUANT=true                      # BNB 4-bit use double quantization

R=8                                                 # Lora attention dimension
LORA_ALPHA=16                                       # Lora alpha parameter
LORA_DROPOUT=0.05                                   # Lora dropout probability
FAN_IN_FAN_OUT=false                                # Lora fan in fan out
BIAS="none"                                         # Lora bias type
TARGET_MODULES="query_key_value"                    # Lora target modules
INFERENCE_MODE=false                                # Inference mode
TASK_TYPE="CAUSAL_LM"                               # Task type

MAX_LENGTH=2048                                     # Max sequence length for tokenizer
TRUNCATION=true                                     # Enable/disable truncation
RETURN_OVERFLOWING_TOKENS=true                      # Return overflowing tokens info
RETURN_LENGTH=true                                  # Return length of encoded inputs
PADDING=true                                        # Enable padding to max sequence length
PADDING_SIDE="right"                                # The side on which the model should have padding applied

MLFLOW_TRACKING_URI="http://localhost:5000"         # URI of MLFlow installed

# Run the script
python LLM_QnA_experiment_main.py \
  --model_name $MODEL_NAME \
  --dataset $DATASET \
  --prompt_type $PROMPT_TYPE \
  --expt_name $EXPT_NAME \
  --cache_dir $CACHE_DIR \
  --output_dir $OUTPUT_DIR \
  --logging_dir $LOGGING_DIR \
  --report_to $REPORT_TO \
  --num_train_epochs $NUM_TRAIN_EPOCHS \
  --train_batch_size $TRAIN_BATCH_SIZE \
  --eval_batch_size $EVAL_BATCH_SIZE \
  --evaluation_strategy $EVALUATION_STRATEGY \
  --eval_steps $EVAL_STEPS \
  --save_steps $SAVE_STEPS \
  --logging_steps $LOGGING_STEPS \
  --learning_rate $LEARNING_RATE \
  --lr_scheduler_type $LR_SCHEDULER_TYPE \
  --optim $OPTIM \
  --warmup_ratio $WARMUP_RATIO \
  --weight_decay $WEIGHT_DECAY \
  --gradient_accumulation_steps $GRADIENT_ACCUMULATION_STEPS \
  --load_best_model_at_end $LOAD_BEST_MODEL_AT_END \
  --fp16 $FP16 \
  --ddp_find_unused_parameters $DDP_FIND_UNUSED_PARAMETERS \
  --early_stopping_patience $EARLY_STOPPING_PATIENCE \
  --load_in_4bit $LOAD_IN_4BIT \
  --bnb_4bit_quant_type $BNB_4BIT_QUANT_TYPE \
  --bnb_4bit_compute_dtype $BNB_4BIT_COMPUTE_DTYPE \
  --bnb_4bit_use_double_quant $BNB_4BIT_USE_DOUBLE_QUANT \
  --r $R \
  --lora_alpha $LORA_ALPHA \
  --lora_dropout $LORA_DROPOUT \
  --fan_in_fan_out $FAN_IN_FAN_OUT \
  --bias $BIAS \
  --target_modules $TARGET_MODULES \
  --inference_mode $INFERENCE_MODE \
  --task_type $TASK_TYPE \
  --max_length $MAX_LENGTH \
  --truncation $TRUNCATION \
  --return_overflowing_tokens $RETURN_OVERFLOWING_TOKENS \
  --return_length $RETURN_LENGTH \
  --padding $PADDING \
  --padding_side $PADDING_SIDE \
  --mlflow_tracking_uri $MLFLOW_TRACKING_URI \
