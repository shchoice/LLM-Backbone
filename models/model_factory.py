from config.entity.training.finetuning_configuration import FinetuningConfiguration
from models.model_manager import ModelManager


class ModelFactory:
    @staticmethod
    def create_model(config: FinetuningConfiguration):
        return ModelManager(config=config)