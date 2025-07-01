from zenml import step

from typing import Tuple, List, Annotated
from documentClassifire.configuration.cofiguration_setup import ConfigManager
from documentClassifire.entity.config_entity import DataIngestionConfig, PreprocessConfig, MlflowConfig
from documentClassifire.src.dataingesstion import DataIngection
from documentClassifire.src.preprocess import create_datagen_from_config
from documentClassifire.src.modelbuild import ModelBuilder
from documentClassifire.connection.connection_mlflow import MlflowConnection
from documentClassifire.src.modeltrainer import ModelTrainer
from tensorflow.keras.models import Model
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from documentClassifire.utils.utility import read_yaml_file





@step
def config_management_step()-> Annotated[ConfigManager, "configuration_manager"]:
  try:
    config=ConfigManager()
    return config
  except Exception as e:
    raise e

@step
def data_ingestion_step(data_ingestion_config)-> Tuple[List[str], List[str], List[str]]:
  try:
    data_ingestion=DataIngection(data_ingection_config=data_ingestion_config)
    return data_ingestion.get_file_path()
  except Exception as e:
    raise e


@step
def preprocess_step(preprocess_config:PreprocessConfig)-> ImageDataGenerator:
  try:
    custom_datagen = create_datagen_from_config(preprocess_config)
    return custom_datagen
  except Exception as e:
    raise e


@step
def cnn_model_build_step(model_build_config)-> Model:
  try:
    model_build=ModelBuilder(model_build_config)
    return model_build.cnn_model()
  except Exception as e:
    raise e

@step
def vgg16_model_build_step(model_build_config)-> Model:
  try:
    model_build=ModelBuilder(model_build_config)
    return model_build.VGG16_model()
  except Exception as e:
    raise e

@step
def EfficientNetB0_model_build_step(model_build_config)-> Model:
  try:
    model_build=ModelBuilder(model_build_config)
    return model_build.EfficientNetB0_model()
  except Exception as e:
    raise e

@step
def mlflow_connection(mlflow_config: MlflowConfig):
  mlf=MlflowConnection(mlflow_config)
  mlf.mlflow_connection()

@step
def model_trainer_step(train_paths:List, val_paths:List, test_paths:List)-> ModelTrainer:
  try:
    model_trainer=ModelTrainer(train_paths, val_paths, test_paths)
    return model_trainer
  except Exception as e:
    raise e
