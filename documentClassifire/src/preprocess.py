from typing import Tuple, List
import numpy as np
import math
import cv2
import os
from keras.utils import Sequence
from sklearn.preprocessing import MultiLabelBinarizer
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from documentClassifire.entity.config_entity import PreprocessConfig
from documentClassifire.logger.logger import setup_logger

logger = setup_logger()

def create_datagen_from_config(config: PreprocessConfig) -> ImageDataGenerator:
    return ImageDataGenerator(
        rotation_range=config.rotation_range,
        width_shift_range=config.width_shift_range,
        height_shift_range=config.height_shift_range,
        shear_range=config.shear_range,
        zoom_range=config.zoom_range,
        horizontal_flip=config.horizontal_flip
    )


class Dataset(Sequence):
    def __init__(self, paths: List[str], batch_size: int, input_shape: Tuple[int, int, int],
                 datagen: ImageDataGenerator = None, augment: bool = False):
        super(Dataset, self).__init__()
        self.batch_size = batch_size
        self.__data = paths
        self.image_size = input_shape
        self.augment = augment
        self.datagen = datagen

        self.classes = []
        self.label2id = {}
        self.id2label = {}
        self.mlb = None  # MultiLabelBinarizer instance

        self.__labels = self.__list_labels(self.__data)

    def __len__(self):
        return math.ceil(len(self.__data) / self.batch_size)

    def __getitem__(self, idx):
        batch_x = self.__data[idx * self.batch_size:(idx + 1) * self.batch_size]
        batch_y = self.__labels[idx * self.batch_size:(idx + 1) * self.batch_size]

        images = np.array([self.__load_image(i) for i in batch_x])
        return images, np.array(batch_y)

    def __list_labels(self, paths: List[str]):
        labels = []
        for path in paths:
            label = path.split(os.path.sep)[-2].split(' ')
            labels.append(label)

        self.mlb = MultiLabelBinarizer()
        labels_bin = self.mlb.fit_transform(labels)
        self.classes = list(self.mlb.classes_)

        # Generate label2id and id2label mappings
        self.label2id = {label: idx for idx, label in enumerate(self.classes)}
        self.id2label = {idx: label for idx, label in enumerate(self.classes)}

        return labels_bin

    """def __load_image(self, path: str):
        img = cv.imread(path)
        img = cv.resize(img, (self.image_size[0], self.image_size[1]))
        img = cv.cvtColor(img, cv.COLOR_BGR2RGB)

        if self.augment:
            img = train_datagen.random_transform(img)

        img = img.astype("float32") / 255.0
        return img"""

    def __load_image(self, path: str):
        img = cv2.imread(path)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        img = cv2.resize(img, (self.image_size[1], self.image_size[0]))  # width, height

        if self.augment:
            img = self.datagen.random_transform(img)

        img = img.astype("float32") / 255.0
        #img = np.reshape(img, self.image_size)  # (height, width, channels)
        return img
