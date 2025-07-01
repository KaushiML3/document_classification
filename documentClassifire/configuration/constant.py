
#from google.colab import userdata
from pathlib import Path

# Data ingestion
TRAIN_DS_PATH:Path=Path('dataset/sample_text_ds/train')
TEST_DS_PATH:Path=Path('dataset/sample_text_ds/test')
TRAIN_SPLIT_SIZE=0.8
## preprocess
PREPROCESS_CONFIG_PATH:Path=Path('config/preprocess_config.yaml')

## Model training
CNN_MODEL_CONFIG_PATH:Path=Path('config/cnn_model_config.yaml')
VGG_MODEL_CONFIG_PATH:Path=Path('config/vgg16_model_config.yaml')
EFF_MODEL_CONFIG_PATH:Path=Path('config/eff_model_config.yaml')

# Model Build
INPUT_SHAPE=(224,224,3)
CLASS_LENGTH=5


##MLFLOW
MLFLOW_TRACKING="https://dagshub.com/kaushigihanml/document_classification.mlflow"
REPO_OWNER='kaushigihanml'
REPO_NAME='document_classification'
