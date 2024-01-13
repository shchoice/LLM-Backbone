class TrainingConfig:
    def __init__(self, num_train_epochs, train_batch_size, eval_batch_size, evaluation_strategy, eval_steps, save_steps,
                 logging_steps, learning_rate, lr_scheduler_type, optim, warmup_ratio, weight_decay,
                 gradient_accumulation_steps, load_best_model_at_end, fp16,
                 ddp_find_unused_parameters, early_stopping_patience):
        self.num_train_epochs = num_train_epochs
        self.train_batch_size = train_batch_size
        self.eval_batch_size = eval_batch_size

        self.evaluation_strategy = evaluation_strategy
        self.eval_steps = eval_steps
        self.save_steps = save_steps
        self.logging_steps = logging_steps

        self.learning_rate = learning_rate
        self.lr_scheduler_type = lr_scheduler_type
        self.optim = optim

        self.warmup_ratio = warmup_ratio
        self.warmup_steps = None

        self.weight_decay = weight_decay

        self.gradient_accumulation_steps = gradient_accumulation_steps
        self.load_best_model_at_end = load_best_model_at_end
        self.fp16 = fp16
        self.ddp_find_unused_parameters = ddp_find_unused_parameters
        self.early_stopping_patience = early_stopping_patience

    def __str__(self):
        return (f"TrainingConfig("
                f"num_train_epochs={self.num_train_epochs}, "
                f"train_batch_size={self.train_batch_size}, "
                f"eval_batch_size={self.eval_batch_size}, "
                f"evaluation_strategy={self.evaluation_strategy}, "
                f"eval_steps={self.eval_steps}, "
                f"save_steps={self.save_steps}, "
                f"logging_steps={self.logging_steps}, "
                f"learning_rate={self.learning_rate}, "
                f"lr_scheduler_type={self.lr_scheduler_type}, "
                f"optim={self.optim}, "
                f"warmup_ratio={self.warmup_ratio}, "
                f"weight_decay={self.weight_decay}, "
                f"gradient_accumulation_steps={self.gradient_accumulation_steps}, "
                f"load_best_model_at_end={self.load_best_model_at_end}, "
                f"fp16={self.fp16}, "
                f"ddp_find_unused_parameters={self.ddp_find_unused_parameters}, "
                f"early_stopping_patience={self.early_stopping_patience})")

    def set_warmup_steps(self, total_sample):
        steps_per_epoch = total_sample / self.train_batch_size
        total_steps = steps_per_epoch * self.num_train_epochs
        self.warmup_steps = int(total_steps * self.warmup_ratio)
