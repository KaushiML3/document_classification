import pandas as pd
from typing import Tuple, List
import os
import cv2
from sklearn.model_selection import train_test_split
from documentClassifire.entity.config_entity import DataIngestionConfig
from pathlib import Path
from documentClassifire.logger.logger import setup_logger

logger = setup_logger()

class DataIngection():
  """
  Data intection class

  """
  def __init__(self,data_ingection_config:DataIngestionConfig):
    """
    Args:data_ingection_config:DataIngectionConfig
    """
    try:
      self.data_ingection_config=data_ingection_config
    except Exception as e:
      logger.error(f"Error initializing DataIngection: {e}")
      raise e

  @staticmethod
  def get_image_paths(path_to_subset:Path) -> List[str]:
  # Collect all valid image paths
    paths = []
    for folder in os.listdir(path_to_subset):
        folder_path = os.path.join(path_to_subset, folder)
        for image in os.listdir(folder_path):
            path_to_image = os.path.join(folder_path, image)

            # Check if image is valid
            img = cv2.imread(path_to_image)
            if img is not None:
                paths.append(path_to_image)
    return paths


  def get_file_path(self) -> Tuple[List[str], List[str], List[str]]:
      """
      Load file paths for train, validation, and test datasets.
      """
      try:
          # Get and split train, val
          paths = DataIngection.get_image_paths(path_to_subset=self.data_ingection_config.train_path)
          train_paths, val_paths = train_test_split(paths, train_size=self.data_ingection_config.train_split_size, shuffle=True, random_state=42)

          # get test paths
          test_paths = DataIngection.get_image_paths(path_to_subset=self.data_ingection_config.test_path)

          return train_paths, val_paths, test_paths

      except Exception as e:
        logger.error(f"Error in get_file_path: {e}")
        raise e



