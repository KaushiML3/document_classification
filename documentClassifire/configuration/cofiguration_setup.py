from documentClassifire.logger.logger import setup_logger

#from google.colab import userdata
from pathlib import Path
from typing import Tuple, List
from documentClassifire.configuration.constant import *
from documentClassifire.entity.config_entity import *
from documentClassifire.utils.utility import read_yaml_file

logger = setup_logger()
class ConfigManager():

  def __init__(self,train_path:Path=TRAIN_DS_PATH,test_path:Path=TEST_DS_PATH,train_split_size:float=TRAIN_SPLIT_SIZE,
               preprocess_congig_path:Path=PREPROCESS_CONFIG_PATH,input_shape:Tuple[int,int,int]=INPUT_SHAPE,
               mlflow_tracking:str=MLFLOW_TRACKING,repo_owner:str=REPO_OWNER,repo_name:str=REPO_NAME,
               class_length:int=CLASS_LENGTH,cnn_model_trainer_config_path:Path=CNN_MODEL_CONFIG_PATH,vgg_model_trainer_config_path:Path=VGG_MODEL_CONFIG_PATH,
               eff_model_trainer_config_path:Path=EFF_MODEL_CONFIG_PATH
               ):
    #data ingestion
    self.train_path=train_path
    self.test_path=test_path
    self.train_split_size=train_split_size

    #Model Build
    self.input_shape=input_shape
    self.class_length=class_length
    
    #MLFlow setup
    self.mlflow_tracking=mlflow_tracking
    self.repo_owner=repo_owner
    self.repo_name=repo_name

    #YAML file path
    self.preprocess_congig_path=preprocess_congig_path
    self.cnn_model_trainer_config_path=cnn_model_trainer_config_path
    self.vgg_model_trainer_config_path=vgg_model_trainer_config_path
    self.eff_model_trainer_config_path=eff_model_trainer_config_path

    # model trainer config
    self.proprocess_config=read_yaml_file(self.preprocess_congig_path)
    self.cnn_model_trainer_config=read_yaml_file(self.cnn_model_trainer_config_path)
    self.vgg_model_trainer_config=read_yaml_file(self.vgg_model_trainer_config_path)
    self.eff_model_trainer_config=read_yaml_file(self.eff_model_trainer_config_path)


  def get_data_ingestion_config(self) -> DataIngestionConfig:
    try:

        data_ingestion_config = DataIngestionConfig(
            train_path=self.train_path,
            test_path=self.test_path,
            train_split_size=self.train_split_size

        )

        return data_ingestion_config
    except Exception as e:
      logger.error(f"Error in get_data_ingestion_config: {e}")
      raise e


  def get_mlflow_config(self) -> MlflowConfig:
    try:
      mlflow_config=MlflowConfig(

      mlflow_tracking=self.mlflow_tracking,
      repo_owner=self.repo_owner,
      repo_name=self.repo_name
      )
      return mlflow_config

    except Exception as e:
      logger.error(f"Error in get_mlflow_config: {e}")
      raise e



  def get_preprocess_config(self) -> PreprocessConfig:
    try:
      config=self.proprocess_config
      preprocess_config=PreprocessConfig(
          rotation_range=config.rotation_range,
          width_shift_range=config.width_shift_range,
          height_shift_range=config.height_shift_range,
          shear_range=config.shear_range,
          zoom_range=config.zoom_range,
          horizontal_flip=config.horizontal_flip
      )

      return preprocess_config

    except Exception as e:
      logger.error(f"Error in get_preprocess_config: {e}")
      raise e

  def get_model_build_config(self) -> ModelBuildConfig:
    try:
      model_build_config=ModelBuildConfig(
          input_shape=self.input_shape,
          len_class=self.class_length
      )
      return model_build_config

    except Exception as e:
      logger.error(f"Error in get_model_build_config: {e}")
      raise e

  def get_cnn_model_trainer_config(self) -> ModelTrainerConfig:
    try:
      config = self.cnn_model_trainer_config
      cnn_model_trainer_config = ModelTrainerConfig(
          experiment_name=config.experiment_name,
          run_name=config.run_name,
          epochs=config.epochs,
          batch_size=config.batch_size,
          learning_rate=config.learning_rate,
          optimizer=config.optimizer,
          loss_function=config.loss_function,
          dropout_rate=config.dropout_rate,
          hidden_units=config.hidden_units,
          input_shape=config.input_shape
      )

      return cnn_model_trainer_config

    except Exception as e:
      logger.error(f"Error in get_cnn_model_trainer_config: {e}")
      raise e



  def get_vgg_model_trainer_config(self) -> ModelTrainerConfig:
    try:
      config = self.vgg_model_trainer_config
      vgg_model_trainer_config = ModelTrainerConfig(
          experiment_name=config.experiment_name,
          run_name=config.run_name,
          epochs=config.epochs,
          batch_size=config.batch_size,
          learning_rate=config.learning_rate,
          optimizer=config.optimizer,
          loss_function=config.loss_function,
          dropout_rate=config.dropout_rate,
          hidden_units=config.hidden_units,
          input_shape=config.input_shape
      )

      return vgg_model_trainer_config

    except Exception as e:
      logger.error(f"Error in get_vgg_model_trainer_config: {e}")
      raise e


  def get_eff_model_trainer_config(self) -> ModelTrainerConfig:
    try:
      config = self.eff_model_trainer_config
      eff_model_trainer_config = ModelTrainerConfig(
          experiment_name=config.experiment_name,
          run_name=config.run_name,
          epochs=config.epochs,
          batch_size=config.batch_size,
          learning_rate=config.learning_rate,
          optimizer=config.optimizer,
          loss_function=config.loss_function,
          dropout_rate=config.dropout_rate,
          hidden_units=config.hidden_units,
          input_shape=config.input_shape
      )

      return eff_model_trainer_config

    except Exception as e:
      logger.error(f"Error in get_eff_model_trainer_config: {e}")
      raise e
