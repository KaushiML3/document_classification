import logging
import joblib
import tensorflow as tf
from pathlib import Path
from sklearn.preprocessing import MultiLabelBinarizer
import cv2
import numpy as np
import logging
import cv2
import keras
from pathlib import Path
import tensorflow as tf
from typing import Optional, Tuple, List
from app.src.logger import setup_logger


# Configure logging
logger = setup_logger("vgg16_load")

def load_vgg_artifacts(model_path: Path, mlb_path: Path) -> tuple[tf.keras.Model, MultiLabelBinarizer]:
    """
    Loads the VGG model and the MultiLabelBinarizer from specified paths.

    Args:
        model_path: Path to the VGG model file (.keras).
        mlb_path: Path to the MultiLabelBinarizer file (.joblib).

    Returns:
        A tuple containing the loaded Keras model and MultiLabelBinarizer object.

    Raises:
        FileNotFoundError: If either the model file or the MLB file is not found.
        Exception: If any other error occurs during loading.
    """
    model = None
    mlb = None
    try:
        logger.info(f"Attempting to load VGG model from {model_path}")
        model = tf.keras.models.load_model(model_path)
        logger.info("VGG model loaded successfully.")
    except FileNotFoundError:
        logger.error(f"Error: VGG model file not found at {model_path}")
        raise
    except Exception as e:
        logger.error(f"An error occurred while loading the VGG model: {e}")
        raise

    try:
        logger.info(f"Attempting to load MultiLabelBinarizer from {mlb_path}")
        mlb = joblib.load(mlb_path)
        logger.info("MultiLabelBinarizer loaded successfully.")
    except FileNotFoundError:
        logger.error(f"Error: MultiLabelBinarizer file not found at {mlb_path}")
        raise
    except Exception as e:
        logger.error(f"An error occurred while loading the MultiLabelBinarizer: {e}")
        raise

    logger.info("Both VGG model and MultiLabelBinarizer loaded successfully.")
    return model, mlb




def preprocess_image(image_path: Path, target_size: tuple[int, int] = (224, 224)) -> np.ndarray | None:
    """
    Preprocesses an image for VGG model prediction.

    Loads an image from the specified path, converts it to RGB, resizes it,
    and normalizes pixel values. Includes robust error handling and logging
    at each step.

    Args:
        image_path: Path to the image file.
        target_size: A tuple (width, height) specifying the desired output size.

    Returns:
        A preprocessed NumPy array representing the image with pixel values
        scaled between 0 and 1, or None if an error occurred during processing.
    """
    try:
        logger.info(f"Attempting to load image from {image_path}")
        img = cv2.imread(str(image_path)) # cv2.imread expects a string or numpy array

        if img is None:
            logger.error(f"Error: Could not load image from {image_path}. cv2.imread returned None.")
            return None
        logger.info("Image loaded successfully.")

        logger.info("Attempting to convert image to RGB.")
        # Check if the image is already in a format that doesn't need BGR to RGB conversion
        # cv2.imread loads in BGR format by default for color images.
        # If the image is grayscale, it might be loaded as such.
        # We want RGB for consistency with models trained on RGB data.
        if len(img.shape) == 3 and img.shape[2] == 3: # Check if it's a color image (likely BGR)
            try:
                img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
                logger.info("Image converted to RGB successfully.")
            except cv2.error as e:
                logger.error(f"Error during BGR to RGB conversion for image {image_path}: {e}")
                return None
        elif len(img.shape) == 2: # Grayscale image
             try:
                img = cv2.cvtColor(img, cv2.COLOR_GRAY2RGB)
                logger.info("Grayscale image converted to RGB successfully.")
             except cv2.error as e:
                logger.error(f"Error during Grayscale to RGB conversion for image {image_path}: {e}")
                return None
        else:
             logger.warning(f"Unexpected image format for {image_path}. Attempting to proceed.")
             # If it's not a standard color or grayscale, we might proceed but log a warning.
             # Depending on requirements, you might want to return None here.


        logger.info(f"Attempting to resize image to {target_size}.")
        try:
            img = cv2.resize(img, target_size)
            if img is None or img.size == 0:
                 logger.error(f"Error: cv2.resize returned None or empty array for image {image_path}.")
                 return None
            logger.info("Image resized successfully.")
        except cv2.error as e:
            logger.error(f"Error during image resizing for image {image_path} to size {target_size}: {e}")
            return None


        logger.info("Attempting to normalize pixel values.")
        try:
            # Ensure the image is the correct dtype before division
            img = img.astype("float32") / 255.0
            if img is None or img.size == 0 or np.max(img) > 1.0 or np.min(img) < 0.0:
                 logger.error(f"Error: Image normalization failed or resulted in unexpected values for image {image_path}.")
                 return None
            logger.info("Pixel values normalized successfully.")
        except Exception as e:
            logger.error(f"Error during pixel normalization for image {image_path}: {e}")
            return None

        logger.info(f"Image preprocessing completed successfully for {image_path}.")
        return img

    except Exception as e:
        logger.error(f"An unexpected error occurred during image preprocessing for {image_path}: {e}")
        return None





class VGGDocumentClassifier:
    """
    A class for classifying documents using a VGG16 model.

    This class encapsulates the loading of the VGG16 model and its associated
    MultiLabelBinarizer, provides a method to preprocess input images, and
    performs document classification predictions.
    """

    def __init__(self, model_path: Path, mlb_path: Path, target_size: Tuple[int, int] = (224, 224)) -> None:
        """
        Initializes the VGGDocumentClassifier by loading the model and MLB.

        Args:
            model_path: Path to the VGG model file (.keras).
            mlb_path: Path to the MultiLabelBinarizer file (.joblib).
            target_size: The target size (width, height) for image preprocessing.
                         Defaults to (224, 224).

        Raises:
            FileNotFoundError: If either the model file or the MLB file is not found.
            Exception: If any other error occurs during loading.
        """
        logger.info("Initializing VGGDocumentClassifier.")
        self.model: Optional[tf.keras.Model] = None
        self.mlb: Optional[MultiLabelBinarizer] = None
        self.target_size: Tuple[int, int] = target_size

        try:
            self._load_artifacts(model_path, mlb_path)
            if self.model and self.mlb:
                logger.info("VGGDocumentClassifier initialized successfully.")
            else:
                logger.critical("VGGDocumentClassifier failed to fully initialize due to artifact loading errors.")
                raise RuntimeError("Failed to load all required artifacts for VGGDocumentClassifier.")
        except Exception as e:
            logger.critical(f"Failed to initialize VGGDocumentClassifier: {e}", exc_info=True)
            raise # Re-raise the exception after logging


    def _load_artifacts(self, model_path: Path, mlb_path: Path) -> None:
        """
        Loads the VGG model and MultiLabelBinarizer with error handling and logging.

        Args:
            model_path: Path to the VGG model file (.keras).
            mlb_path: Path to the MultiLabelBinarizer file (.joblib).

        Raises:
            FileNotFoundError: If either the model file or the MLB file is not found.
            Exception: If any other unexpected error occurs during loading.
        """
        logger.info("Starting artifact loading.")
        model_loaded: bool = False
        mlb_loaded: bool = False

        # Load Model
        try:
            logger.info(f"Attempting to load VGG model from {model_path}")
            self.model = tf.keras.models.load_model(model_path)
            logger.info("VGG model loaded successfully.")
            model_loaded = True
        except FileNotFoundError:
            logger.critical(f"Critical Error: VGG model file not found at {model_path}", exc_info=True)
            raise # Re-raise to indicate a critical initialization failure
        except Exception as e:
            logger.critical(f"Critical Error: An unexpected error occurred while loading the VGG model from {model_path}: {e}", exc_info=True)
            raise # Re-raise to indicate a critical initialization failure

        # Load MLB
        try:
            logger.info(f"Attempting to load MultiLabelBinarizer from {mlb_path}")
            self.mlb = joblib.load(mlb_path)
            logger.info("MultiLabelBinarizer loaded successfully.")
            mlb_loaded = True
        except FileNotFoundError:
            logger.critical(f"Critical Error: MultiLabelBinarizer file not found at {mlb_path}", exc_info=True)
            raise # Re-raise to indicate a critical initialization failure
        except Exception as e:
            logger.critical(f"Critical Error: An unexpected error occurred while loading the MultiLabelBinarizer from {mlb_path}: {e}", exc_info=True)
            raise # Re-raise to indicate a critical initialization failure

        if model_loaded and mlb_loaded:
             logger.info("All required VGG artifacts loaded successfully.")
        else:
            logger.error("One or more required VGG artifacts failed to load during _load_artifacts.")


    def preprocess_image(self, image_path: Path) -> Optional[np.ndarray]:
        """
        Preprocesses an image for VGG model prediction.

        Loads an image from the specified path, converts it to RGB, resizes it,
        and normalizes pixel values. Includes robust error handling and logging
        at each step.

        Args:
            image_path: Path to the image file.

        Returns:
            A preprocessed NumPy array representing the image with pixel values
            scaled between 0 and 1, or None if an error occurred during processing.
        """
        try:
            logger.info(f"Attempting to load image from {image_path}")
            img = cv2.imread(str(image_path)) # cv2.imread expects a string or numpy array

            if img is None:
                logger.error(f"Error: Could not load image from {image_path}. cv2.imread returned None.")
                return None
            logger.info("Image loaded successfully.")

            logger.info("Attempting to convert image to RGB.")
            if len(img.shape) == 3 and img.shape[2] == 3: # Check if it's a color image (likely BGR)
                try:
                    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
                    logger.info("Image converted to RGB successfully.")
                except cv2.error as e:
                    logger.error(f"Error during BGR to RGB conversion for image {image_path}: {e}")
                    return None
            elif len(img.shape) == 2: # Grayscale image
                 try:
                    img = cv2.cvtColor(img, cv2.COLOR_GRAY2RGB)
                    logger.info("Grayscale image converted to RGB successfully.")
                 except cv2.error as e:
                    logger.error(f"Error during Grayscale to RGB conversion for image {image_path}: {e}")
                    return None
            else:
                 logger.warning(f"Unexpected image format for {image_path}. Attempting to proceed.")


            logger.info(f"Attempting to resize image to {self.target_size}.")
            try:
                img = cv2.resize(img, self.target_size)
                if img is None or img.size == 0:
                     logger.error(f"Error: cv2.resize returned None or empty array for image {image_path}.")
                     return None
                logger.info("Image resized successfully.")
            except cv2.error as e:
                logger.error(f"Error during image resizing for image {image_path} to size {self.target_size}: {e}")
                return None


            logger.info("Attempting to normalize pixel values.")
            try:
                img = img.astype("float32") / 255.0
                if img is None or img.size == 0 or np.max(img) > 1.0 or np.min(img) < 0.0:
                     logger.error(f"Error: Image normalization failed or resulted in unexpected values for image {image_path}.")
                     return None
                logger.info("Pixel values normalized successfully.")
            except Exception as e:
                logger.error(f"Error during pixel normalization for image {image_path}: {e}")
                return None

            logger.info(f"Image preprocessing completed successfully for {image_path}.")
            return img

        except Exception as e:
            logger.error(f"An unexpected error occurred during image preprocessing for {image_path}: {e}")
            return None


    def predict(self, image_path: Path) -> Optional[List[str]]:
        """
        Predicts the class labels for a given image using the loaded VGG model.

        The process involves loading and preprocessing the image, performing
        inference with the model, and converting the prediction to class labels
        using the MultiLabelBinarizer.

        Args:
            image_path: Path to the image file to classify.

        Returns:
            A list of predicted class labels (strings) if the prediction process
            is successful. Returns None if any critical step (image loading,
            preprocessing, model inference, or inverse transform) fails.
            Returns an empty list if the prediction process is successful but
            no labels are predicted.
        """
        logger.info(f"Starting prediction process for image: {image_path}.")

        if self.model is None or self.mlb is None:
            logger.error("Model or MultiLabelBinarizer not loaded. Cannot perform prediction.")
            return None

        # Preprocess image
        image = self.preprocess_image(image_path)
        if image is None:
            logger.error(f"Image preprocessing failed for {image_path}. Cannot perform prediction.")
            return None

        try:
            logger.info(f"Performing model inference for {image_path}.")
            # Add batch dimension to the image
            image = np.expand_dims(image, axis=0)
            prd = self.model.predict(image)
            logger.info(f"Model inference completed for {image_path}. Prediction shape: {prd.shape}")
        except Exception as e:
            logger.error(f"An error occurred during model inference for {image_path}: {e}", exc_info=True)
            return None


        # Convert the prediction to a binary indicator format and get labels
        try:
            logger.info(f"Converting prediction to labels for {image_path}.")
            # Assuming multi-class classification for now, taking the argmax
            # If it's multi-label, you'd apply a sigmoid and thresholding here
            pred_id = np.argmax(prd, axis=1)

            # Create a zero array with the shape (1, number of classes)
            binary_prediction = np.zeros((1, len(self.mlb.classes_)))
            # Set the index of the predicted class to 1
            binary_prediction[0, pred_id] = 1


            predicted_labels_tuple_list: List[Tuple[str, ...]] = self.mlb.inverse_transform(binary_prediction)
            logger.info(f"Prediction processed for {image_path}. Predicted labels (raw tuple list): {predicted_labels_tuple_list}")

            if predicted_labels_tuple_list and len(predicted_labels_tuple_list) > 0:
                 final_labels: List[str] = list(predicted_labels_tuple_list[0])
                 logger.info(f"Final predicted labels for {image_path}: {final_labels}")
                 return final_labels
            else:
                 logger.warning(f"MLB inverse_transform returned an empty list for {image_path}. No labels predicted.")
                 return []

        except Exception as e:
            logger.error(f"An error occurred during inverse transform or label processing for {image_path}: {e}", exc_info=True)
            return None
        

