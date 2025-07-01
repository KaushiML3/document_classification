from tensorflow.keras import Model
from tensorflow.keras.applications import EfficientNetB0
from tensorflow.keras import layers, models
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Dropout, Dense, BatchNormalization, Flatten, GlobalAveragePooling2D
from documentClassifire.entity.config_entity import ModelBuildConfig
from tensorflow.keras.applications import VGG16
import tensorflow as tf
from documentClassifire.logger.logger import setup_logger

logger = setup_logger()

class ModelBuilder():
  def __init__(self,model_build_config:ModelBuildConfig):
    self.model_build_config=model_build_config
    self.input_shape=model_build_config.input_shape
    self.len_class=model_build_config.len_class


  def cnn_model(self) -> Model:
      model = tf.keras.models.Sequential()

      model.add(Conv2D(32, 3, padding = 'same', input_shape = self.input_shape, kernel_initializer = 'he_normal', activation = 'relu'))
      model.add(BatchNormalization()) #Stack of images become stack with no negative values
      model.add(MaxPooling2D(2))
      model.add(Dropout(0.25))

      model.add(Conv2D(64, 3, padding = 'same', kernel_initializer = 'he_normal', activation = 'relu'))
      model.add(BatchNormalization())
      model.add(Conv2D(64, 3, padding = 'same', kernel_initializer = 'he_normal', activation = 'relu'))
      model.add(BatchNormalization())
      model.add(MaxPooling2D())
      model.add(Dropout(0.25))

      model.add(Conv2D(128, 3, padding = 'same', kernel_initializer = 'he_normal', activation = 'relu'))
      model.add(BatchNormalization())
      model.add(Conv2D(128, 2, padding = 'same', kernel_initializer = 'he_normal', activation = 'relu'))
      model.add(BatchNormalization())
      model.add(MaxPooling2D())
      model.add(Dropout(0.25))

      model.add(Conv2D(256, 3, padding = 'same', kernel_initializer = 'he_normal', activation = 'relu'))
      model.add(BatchNormalization())
      model.add(Conv2D(256, 2, padding = 'same', kernel_initializer = 'he_normal', activation = 'relu'))
      model.add(BatchNormalization())
      model.add(MaxPooling2D())
      model.add(Dropout(0.25))

      #model.add(Conv2D(1024, 3, padding = 'same', kernel_initializer = 'he_normal', activation = 'relu'))
      #model.add(BatchNormalization())
      #model.add(Conv2D(1024, 2, padding = 'same', kernel_initializer = 'he_normal', activation = 'relu'))
      #model.add(BatchNormalization())
      #model.add(MaxPooling2D())
      #model.add(Dropout(0.25))

      model.add(Flatten())
      model.add(Dense(1024, activation = 'relu', kernel_initializer = 'he_normal', ))
      model.add(BatchNormalization())
      model.add(Dropout(0.5))

      model.summary()
      logger.info("CNN model summary printed.")
      model.add(Dense(self.len_class, activation = 'softmax'))
      return model

  def VGG16_model(self)-> Model:
      base_model = VGG16(weights='imagenet', include_top=False, input_shape=self.input_shape)
      # Add custom layers for classification with dropout
      x = base_model.output
      x = GlobalAveragePooling2D()(x)
      x = Dense(1024, activation="relu",kernel_initializer = 'he_normal')(x)
      x = BatchNormalization()(x)
      x = Dropout(0.5)(x)

      output = Dense(self.len_class, activation="sigmoid")(x)
      # Create the complete model
      model = Model(inputs=base_model.input, outputs=output)
      # Summary of the model
      model.summary()
      logger.info("VGG16 model summary printed.")

      return model

  def EfficientNetB0_model(self)-> Model:
      base_model = EfficientNetB0(weights='imagenet', include_top=False, input_shape=self.input_shape)

      # Freeze base model
      for layer in base_model.layers:
          layer.trainable = False

      model = models.Sequential()
      model.add(base_model)
      model.add(layers.GlobalAveragePooling2D())
      model.add(layers.Dense(512, activation='relu',kernel_initializer = 'he_normal', name='dense'))
      model.add(layers.Dropout(0.5))
      model.add(layers.Dense(self.len_class, activation='softmax', name='predictions'))

      model.summary()
      logger.info("EfficientNetB0 model summary printed.")
      return model
