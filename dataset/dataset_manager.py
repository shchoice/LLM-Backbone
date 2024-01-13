import transformers
from transformers import DataCollatorForLanguageModeling

from config.entity.models.model_config import ModelConfig
from config.entity.training.finetuning_configuration import FinetuningConfiguration
from dataset.supervised_dataset import SupervisedDataset


class DatasetManager:
    def __init__(self, config: FinetuningConfiguration):
        self.dataset = None
        self.prompt = None

        self.model_config: ModelConfig = config.model_config

    def get_prompt_type(self):
        pass

    def make_supervised_dataset(self, tokenizer: transformers.PreTrainedTokenizer):
        train_dataset = SupervisedDataset(tokenizer=tokenizer, data_path=self.model_config.dataset_path)
        data_collator = DataCollatorForLanguageModeling(tokenizer=tokenizer, mlm=False)

        return dict(
            train_dataset=train_dataset,
            eval_dataset = None,
            data_collator=data_collator,
        )
