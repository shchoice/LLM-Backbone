from typing import Dict, Sequence

import transformers
from transformers import AutoTokenizer

from config import constants
from config.entity.models.model_config import ModelConfig
from config.entity.tokenizer.tokenizer_config import TokenizerConfig
from config.entity.training.finetuning_configuration import FinetuningConfiguration
from config.entity.training.training_logging_config import TrainingLoggingConfig


class TokenizerManager:
    def __init__(self, config: FinetuningConfiguration):
        model_config: ModelConfig = config.model_config
        self.model_name = model_config.model_name
        self.model = None

        tokenizer_config: TokenizerConfig = config.tokenizer_config
        self.tokenizer_config = tokenizer_config
        self.tokenizer: transformers.PreTrainedTokenizer = None

        trainer_logging_config: TrainingLoggingConfig = config.trainer_logging_config
        self.cache_dir = trainer_logging_config.cache_dir

    def load_tokenizer(self):
        self.tokenizer = AutoTokenizer.from_pretrained(
            self.model_name,
            cache_dir=self.cache_dir,
            padding_side=self.tokenizer_config.padding_side,
        )
        self.tokenizer.pad_token = self.tokenizer.eos_token
        print(
            f'Check bos_token, pad_token, eos_tokenm unk_token : {self.tokenizer.bos_token}, '
            f'{self.tokenizer.pad_token}, {self.tokenizer.eos_token}, {self.tokenizer.unk_token}')
        return self.tokenizer

    def set_model(self, model):
        self.model = model

    def _tokenize_fn(self, strings: Sequence[str], tokenizer: transformers.PreTrainedTokenizer) -> Dict:
        tokenized_list = [
            self.tokenizer(
                text,
                return_tensors="pt",
                padding="longest",
                max_length=self.tokenizer.model_max_length,
                truncation=True,
            )
            for text in strings
        ]
        input_ids = labels = [tokenized.input_ids[0] for tokenized in tokenized_list]
        input_ids_lens = labels_lens = [
            tokenized.input_ids.ne(tokenizer.pad_token_id).sum().item() for tokenized in tokenized_list
        ]
        return dict(
            input_ids=input_ids,
            labels=labels,
            input_ids_lens=input_ids_lens,
            labels_lens=labels_lens,
        )

    def get_model_tokens(self) -> Dict[str, str]:
        special_tokens_dict:Dict = dict(
            bos_token=constants.DEFAULT_BOS_TOKEN,
            eos_token=constants.DEFAULT_EOS_TOKEN,
            unk_token=constants.DEFAULT_UNK_TOKEN,
            pad_token=constants.DEFAULT_PAD_TOKEN,
        )
        if self.model_name == constants.MODEL_NAME_LLAMA_2_7B:
            special_tokens_dict['bos_token'] = constants.LLAMA2_BOS_TOKEN
            special_tokens_dict['eos_token'] = constants.LLAMA2_EOS_TOKEN
            special_tokens_dict['unk_token'] = constants.LLAMA2_UNK_TOKEN
            special_tokens_dict['pad_token'] = constants.LLAMA2_PAD_TOKEN

        return special_tokens_dict

    def add_special_token(self):
        special_tokens_dict = self.get_model_tokens()

        pad_token = dict(pad_token=special_tokens_dict['pad_token'])
        if self.tokenizer.pad_token is None:
            self.smart_tokenizer_and_embedding_resize(
                special_tokens_dict=special_tokens_dict['pad_token'],
                tokenizer=self.tokenizer,
                model=self.model,
            )

    def smart_tokenizer_and_embedding_resize(self,
                                             special_tokens_dict: Dict,
                                             tokenizer: transformers.PreTrainedTokenizer,
                                             model: transformers.PreTrainedModel):
        # 특수 토큰을 토크나이저에 추가하고, 추가된 토큰의 수를 반환
        num_new_tokens = tokenizer.add_special_tokens(special_tokens_dict)
        # 모델의 토큰 임베딩 크기를 새로운 토크나이저의 크기에 맞게 조정
        model.resize_token_embeddings(len(tokenizer))

        # 새로운 토큰이 추가되었다면, 추가된 토큰에 대한 임베딩을 처리
        if num_new_tokens > 0:
            # 모델의 입력과 출력 임베딩을 가져옴
            input_embeddings = model.get_input_embeddings().weight.data
            output_embeddings = model.get_output_embeddings().weight.data

            # 기존 토큰의 입력 및 출력 임베딩의 평균을 계산
            input_embeddings_avg = input_embeddings[:-num_new_tokens].mean(dim=0, keepdim=True)
            output_embeddings_avg = output_embeddings[:-num_new_tokens].mean(dim=0, keepdim=True)

            # 새로운 토큰의 임베딩을 기존 토큰 임베딩의 평균으로 설정
            input_embeddings[-num_new_tokens:] = input_embeddings_avg
            output_embeddings[-num_new_tokens:] = output_embeddings_avg
