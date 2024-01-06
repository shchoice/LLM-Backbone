class MLFlowConfig:
    def __init__(self, mlflow_tracking_uri):
        self.mlflow_tracking_uri = mlflow_tracking_uri

    def __str__(self):
        return f"MLFlowConfig(" \
               f"mlflow_tracking_uri={self.mlflow_tracking_uri})"
