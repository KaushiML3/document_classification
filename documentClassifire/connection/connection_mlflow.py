import dagshub
import mlflow
from documentClassifire.entity.config_entity import MlflowConfig



class MlflowConnection():
  def __init__(self,mlflow_config:MlflowConfig):
    self.mlflow_config=mlflow_config

  def mlflow_connection(self):
    dagshub.init(repo_owner=self.mlflow_config.repo_owner, repo_name=self.mlflow_config.repo_name, mlflow=True)
    mlflow.set_tracking_uri(self.mlflow_config.mlflow_tracking)
