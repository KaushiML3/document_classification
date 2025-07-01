from pathlib import Path
from app.src.vgg16_load import VGGDocumentClassifier
from app.src.vit_load import VITDocumentClassifier
from app.src.constant import *
from app.src.logger import setup_logger

logger = setup_logger("model_loader")


def vit_loader()->VITDocumentClassifier:
    try:
        vit=VITDocumentClassifier(vit_model_path, vit_mlb_path)
        return vit
    except Exception as e:
        logger.error(str(e))
        raise e


def vgg_loader():
    try:
        vgg=VGGDocumentClassifier(vgg_model_path, vgg_mlb_path)
        return vgg
    except Exception as e:
        logger.error(str(e))
        raise e
