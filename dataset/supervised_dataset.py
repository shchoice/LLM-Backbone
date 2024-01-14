import copy
from typing import Dict, Sequence

import transformers
from torch.utils.data import Dataset

import utils
from config import constants
from prompts.prompt_factory import PromptFactory


class SupervisedDataset(Dataset):
    def __init__(self, data_path: str, tokenizer: transformers.PreTrainedTokenizer, prompt_type: str):
        super().__init__()

        self.tokenizer = tokenizer
        PROMPT_DICT = PromptFactory.get_prompt(prompt_type=prompt_type)

        list_data_dict = utils.jload(data_path)
        prompt_input, prompt_no_input = PROMPT_DICT["prompt_input"], PROMPT_DICT["prompt_no_input"]

        sources = [
            prompt_input.format(example) if example.get('input', '') else prompt_no_input.format(example) for example in list_data_dict
        ]
        targets = [
            f"{example['output']}{tokenizer.eos_token}" for example in list_data_dict
        ]

        data_dict = self.preprocess(sources, targets, tokenizer)

        self.input_ids = data_dict["input_ids"]
        self.labels = data_dict["labels"]

    def preprocess(
            self,
            sources: Sequence[str],
            targets: Sequence[str],
            tokenizer: transformers.PreTrainedTokenizer) -> Dict:
        examples = [source + target for source, target in zip(sources, targets)]
        examples_tokenized, sources_tokenized = [self.tokenizer._tokenize_fn(strings=strings, tokenizer=tokenizer)
                                                 for strings in (examples, sources)]
        input_ids = examples_tokenized["input_ids"]
        labels = copy.deepcopy(input_ids)
        for label, source_len in zip(labels, sources_tokenized["input_ids_lens"]):
            label[:source_len] = constants.IGNORE_INDEX

        return dict(input_ids=input_ids, labels=labels)
