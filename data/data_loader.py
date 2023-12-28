from datasets import load_dataset

from prompts.prompt_loader import PromptLoader


class DataLoader:
    def __init__(self, dataset, prompt_name, cache_dir):
        self.dataset = self.get_dataset(dataset=dataset)
        self.prompt = PromptLoader.get_prompt(prompt_type=prompt_name)
        self.cache_dir = cache_dir

    def get_dataset(self, dataset):
        if dataset == 'KorQuAD-v1':
            return 'squad_kor_v1'

    def load_and_format_dataset(self):
        self.dataset = load_dataset(self.dataset, cache_dir=self.cache_dir)
        self.dataset = self.dataset.map(self.format_to_prompt)

        return self.dataset

    def format_to_prompt(self, data):
        context = data['context']
        question = data['question']
        answers = f"{data['answers']['text'][0]}"
        formatted_prompt = self.prompt['context_question'].format(context=context, question=question, answers=answers)

        return {"input": formatted_prompt}
