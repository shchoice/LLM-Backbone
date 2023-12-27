class LoraConfiguration:
    def __init__(self, r, lora_alpha, lora_dropout, fan_in_fan_out, bias, target_modules, inference_mode, task_type):
        self.r = r
        self.lora_alpha = lora_alpha
        self.lora_dropout = lora_dropout
        self.fan_in_fan_out = fan_in_fan_out
        self.bias = bias
        self.target_modules = target_modules
        self.inference_mode = inference_mode
        self.task_type = task_type

    def __str__(self):
        return (f"LoraConfiguration("
                f"r={self.r}, "
                f"lora_alpha={self.lora_alpha}, "
                f"lora_dropout={self.lora_dropout}, "
                f"fan_in_fan_out={self.fan_in_fan_out}, "
                f"bias={self.bias}, "
                f"target_modules={self.target_modules}, "
                f"inference_mode={self.inference_mode}, "
                f"task_type={self.task_type})")
