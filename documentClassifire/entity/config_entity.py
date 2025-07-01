from typing import Tuple,List
from documentClassifire.configuration.constant import *
from dataclasses import dataclass
from pathlib import Path
from documentClassifire.logger.logger import setup_logger

logger = setup_logger()
logger.info("This is an info message from config_entity.")
@dataclass(frozen=True)
class DataIngestionConfig:
  """
  Data intection config
  """
  train_path:Path
  test_path:Path
  train_split_size:float


@dataclass(frozen=True)
class PreprocessConfig:
  """
  Data preprocess config
  """
  rotation_range:int
  width_shift_range:float
  height_shift_range:float
  shear_range:float
  zoom_range:float
  horizontal_flip:bool

@dataclass(frozen=True)
class ModelBuildConfig:
  """
  Model build config
  """
  input_shape:Tuple[int,int,int]
  len_class:int



@dataclass(frozen=True)
class ModelTrainerConfig:
  """
  Model build config
  """
  experiment_name:str
  run_name:str
  epochs:int
  batch_size:int
  learning_rate:float
  optimizer:str
  loss_function:str
  dropout_rate:float
  hidden_units:List[int]
  input_shape:list


@dataclass(frozen=True)
class MlflowConfig:
  """
  mlflow config
  """
  mlflow_tracking:str
  repo_owner:str
  repo_name:str
