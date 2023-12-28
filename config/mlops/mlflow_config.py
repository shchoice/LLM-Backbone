class MLFlowConfig:
    def __init__(self, mlflow_tracking_uri, mlflow_port):
        self.mlflow_tracking_uri = mlflow_tracking_uri
        self.mlflow_port = mlflow_port

    def __str__(self):
        return f"MLFlowConfig(" \
               f"mlflow_tracking_uri={self.mlflow_tracking_uri}, " \
               f"mlflow_port={self.mlflow_port})"
