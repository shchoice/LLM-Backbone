class TokenizerConfig:
    def __init__(self, max_length, truncation, return_overflowing_tokens, return_length, padding):
        self.max_length = max_length
        self.truncation = truncation
        self.return_overflowing_tokens = return_overflowing_tokens
        self.return_length = return_length
        self.padding = padding

    def __str__(self):
        return (f"TokenizerConfig("
                f"max_length={self.max_length}, "
                f"truncation={self.truncation}, "
                f"return_overflowing_tokens={self.return_overflowing_tokens}, "
                f"return_length={self.return_length}, "
                f"padding={self.padding})")
