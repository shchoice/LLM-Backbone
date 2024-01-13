import mlflow


class MLFLowManager:
    def set_log_model(self, model, model_name):
        mlflow.pytorch.log_model(model, model_name)

    def set_tag_name(self, tag, tag_contents):
        mlflow.set_tag()

    def set_log_param(self, temp01, temp02):
        mlflow.log_param(temp01, temp02)
