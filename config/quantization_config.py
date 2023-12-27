class QuantizationConfig():
    def __init__(self, load_in_4bit, bnb_4bit_quant_type, bnb_4bit_compute_dtype, bnb_4bit_use_double_quant):
        self.load_in_4bit = load_in_4bit
        self.bnb_4bit_quant_type = bnb_4bit_quant_type
        self.bnb_4bit_compute_dtype = bnb_4bit_compute_dtype
        self.bnb_4bit_use_double_quant = bnb_4bit_use_double_quant

    def __str__(self):
        return (f"QuantizationConfig("
                f"load_in_4bit={self.load_in_4bit}, "
                f"bnb_4bit_quant_type={self.bnb_4bit_quant_type}, "
                f"bnb_4bit_compute_dtype={self.bnb_4bit_compute_dtype}, "
                f"bnb_4bit_use_double_quant={self.bnb_4bit_use_double_quant})")
