class PromptLoader:
    @staticmethod
    def get_prompt(prompt_type):
        if prompt_type == 'A':
            prompt = {
                "context_question": (
                    "Below is an instruction that describes a task. Write a response that appropriately completes the request. "
                    "### Context(맥락):\n{context}\n### Question(질문):\n{question}\n"
                ),
                "answers": (
                    "### Answer(답변):\n{answer}"
                )
            }
            return prompt
        elif prompt_type == 'B':
            prompt = {
                "context_question": (
                    "Below is an instruction that describes a task. Write a response that appropriately completes the request. "
                    "### Question(질문):\n{question}\n"
                ),
                "answers": (
                    "### Answer(답변):\n{answer}"
                )
            }
            return prompt
        elif prompt_type == 'C':
            prompt = {
                'text': (
                    "Below is an instruction that describes a task. Write a response that appropriately completes the request. " \
                    "### Instruction: {question} ### Response: {answer}"
                )
            }
            return prompt
        elif prompt_type == 'D':
            prompt = {
                'text': (
                    "Below is an instruction that describes a task. Write a response that appropriately completes the request. " \
                    "### Instruction: {question} {context} ### Response: {answer}"
                )
            }
            return prompt
        elif prompt_type == 'E':
            prompt = {
                'text': (
                    "Below is an instruction that describes a task. Write a response that appropriately completes the request. " \
                    "### Instruction: {context} {question} ### Response: {answer}"
                )
            }
            return prompt
