from datasets import load_dataset

from prompts.prompt_loader import PromptLoader


class DataLoader:
    def __init__(self, dataset, prompt_type, cache_dir):
        self.dataset = self.get_dataset(dataset=dataset)
        self.prompt = PromptLoader.get_prompt(prompt_type=prompt_type)
        self.prompt_type = prompt_type
        self.cache_dir = cache_dir

    def get_dataset(self, dataset):
        if dataset == 'KorQuAD-v1':
            return 'squad_kor_v1'

    def load_and_format_dataset(self):
        self.dataset = load_dataset(self.dataset, cache_dir=self.cache_dir)
        self.dataset = self.dataset.map(self.format_to_prompt)

        return self.dataset

    def format_to_prompt(self, data):
        if self.prompt_type == 'A' or self.prompt_type == 'B':
            context = data['context']
            question = data['question']
            answer = f"{data['answers']['text'][0]}"
            formatted_prompt = self.prompt['context_question'].format(context=context, question=question, answer=answer)
        elif self.prompt_type == 'C':
            question = data['question']
            answer = f"{data['answers']['text'][0]}"
            formatted_prompt = self.prompt['text'].format(question=question, answer=answer)
        elif self.prompt_type == 'D' or self.prompt_type == 'E':
            context = data['context']
            question = data['question']
            answer = f"{data['answers']['text'][0]}"
            formatted_prompt = self.prompt['text'].format(context=context, question=question, answer=answer)

        return {"input": formatted_prompt}
