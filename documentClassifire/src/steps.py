from zenml import step
from typing import Tuple, List, Annotated
from documentClassifire.configuration.cofiguration_setup import ConfigManager
from documentClassifire.entity.config_entity import (
    DataIngestionConfig,
    PreprocessConfig,
    MlflowConfig,
    ModelBuildConfig,
    ModelTrainerConfig, 

)
from documentClassifire.src.dataingesstion import DataIngection
from documentClassifire.src.preprocess import create_datagen_from_config
from documentClassifire.src.modelbuild import ModelBuilder
from documentClassifire.connection.connection_mlflow import MlflowConnection
from documentClassifire.src.modeltrainer import ModelTrainer
from tensorflow.keras.models import Model
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from documentClassifire.logger.logger import setup_logger

logger = setup_logger()

# --- Core Steps ---
@step
def config_management_step() -> ConfigManager:
    try:
        return ConfigManager()
    except Exception as e:
        logger.error(f"Error in config_management_step: {e}")
        raise e

@step
def data_ingestion_step(config: ConfigManager) -> Tuple[List[str], List[str], List[str]]:
    try:
        data_ingestion = DataIngection(config.get_data_ingestion_config())
        train_paths, val_paths, test_paths=data_ingestion.get_file_path()
        return train_paths, val_paths, test_paths
    except Exception as e:
        logger.error(f"Error in data_ingestion_step: {e}")
        raise e

@step
def preprocess_step(config: ConfigManager) -> ImageDataGenerator:
    try:
        return create_datagen_from_config(config.get_preprocess_config())
    except Exception as e:
        logger.error(f"Error in preprocess_step: {e}")
        raise e

@step
def mlflow_connection_step(config: ConfigManager) -> None:
    try:
        mlf = MlflowConnection(config.get_mlflow_config())
        mlf.mlflow_connection()
    except Exception as e:
        logger.error(f"Error in mlflow_connection_step: {e}")
        raise e

@step
def cnn_model_build_step(config: ConfigManager) -> Model:
    try:
        return ModelBuilder(config.get_model_build_config()).cnn_model()
    except Exception as e:
        logger.error(f"Error in cnn_model_build_step: {e}")
        raise e

@step
def vgg16_model_build_step(config: ConfigManager) -> Model:
    try:
        return ModelBuilder(config.get_model_build_config()).VGG16_model()
    except Exception as e:
        logger.error(f"Error in vgg16_model_build_step: {e}")
        raise e

@step
def efficient_net_b0_model_build_step(config: ConfigManager) -> Model:
    try:
        return ModelBuilder(config.get_model_build_config()).EfficientNetB0_model()
    except Exception as e:
        logger.error(f"Error in efficient_net_b0_model_build_step: {e}")
        raise e

@step
def train_model_step(
    model: Model,
    trainer_config: ModelTrainerConfig,
    datagen: ImageDataGenerator,
    train_paths: List[str],
    val_paths: List[str],
    test_paths: List[str]
) -> None:
    try:
        trainer = ModelTrainer(train_paths, val_paths, test_paths)
        trainer.model_trainer(model, trainer_config, datagen)
    except Exception as e:
        logger.error(f"Error in train_model_step: {e}")
        raise e
    
@step
def get_cnn_model_trainer_config_step(config: ConfigManager) -> ModelTrainerConfig:
    return config.get_cnn_model_trainer_config()

@step
def get_vgg_model_trainer_config_step(config: ConfigManager) -> ModelTrainerConfig:
    return config.get_vgg_model_trainer_config()

@step
def get_eff_model_trainer_config_step(config: ConfigManager) -> ModelTrainerConfig:
    return config.get_eff_model_trainer_config()
