import numpy as np
import torch
from PIL import Image
from transformers import AutoImageProcessor, AutoModelForImageClassification
from sklearn.preprocessing import MultiLabelBinarizer
import joblib
from pathlib import Path
from typing import List, Optional, Tuple, Any
from app.src.logger import setup_logger



logger = setup_logger("vit_load")

class VITDocumentClassifier:
    """
    A class for classifying documents using a Vision Transformer (ViT) model.

    This class encapsulates the loading of the ViT model, its associated processor,
    and a MultiLabelBinarizer for converting model outputs to meaningful labels.
    It provides a method to preprocess input images and perform multi-label
    classification predictions with a specified confidence cutoff threshold.
    """

    def __init__(self, model_path: Path, mlb_path: Path, model_id: str = "google/vit-base-patch16-224-in21k") -> None:
        """
        Initializes the VITDocumentClassifier by loading the model, processor, and MLB.

        Args:
            model_path: Path to the ViT model file (.pth). This is expected to be
                        a pre-trained or fine-tuned PyTorch model file.
            mlb_path: Path to the MultiLabelBinarizer file (.joblib). This file
                      should contain the fitted binarizer object corresponding
                      to the model's output classes.
            model_id: The Hugging Face model ID for the processor. This is used
                      to load the appropriate image processor for the ViT model.
                      Defaults to "google/vit-base-patch16-224-in21k".

        Raises:
            FileNotFoundError: If either the model file or the MLB file is not found
                             at the specified paths during artifact loading.
            Exception: If any other unexpected error occurs during the loading
                       of the model, processor, or MultiLabelBinarizer.
            RuntimeError: If artifact loading fails for critical components
                          (model or MLB).
        """
        logger.info("Initializing VITDocumentClassifier.")
        self.model: Optional[torch.nn.Module] = None
        self.processor: Optional[AutoImageProcessor] = None
        self.mlb: Optional[MultiLabelBinarizer] = None
        self.device: torch.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        logger.info(f"Using device: {self.device}")
        self.model_id: str = model_id

        try:
            self._load_artifacts(model_path, mlb_path)
            if self.model and self.processor and self.mlb:
                logger.info("VITDocumentClassifier initialized successfully.")
            else:
                # This case should ideally be caught and re-raised in _load_artifacts
                # but adding a check here for robustness.
                logger.critical("VITDocumentClassifier failed to fully initialize due to artifact loading errors.")
                raise RuntimeError("Failed to load all required artifacts for VITDocumentClassifier.")

        except Exception as e:
            logger.critical(f"Failed to initialize VITDocumentClassifier: {e}", exc_info=True)
            # Re-raise the exception after logging
            raise


    def _load_artifacts(self, model_path: Path, mlb_path: Path) -> None:
        """
        Loads the ViT model, processor, and MultiLabelBinarizer with enhanced error handling and logging.

        This is an internal helper method called during initialization.

        Args:
            model_path: Path to the ViT model file (.pth).
            mlb_path: Path to the MultiLabelBinarizer file (.joblib).

        Raises:
            FileNotFoundError: If either the model file or the MLB file is not found.
            Exception: If any other unexpected error occurs during loading.
        """
        logger.info("Starting artifact loading.")
        processor_loaded: bool = False
        model_loaded: bool = False
        mlb_loaded: bool = False

        # Load Processor
        try:
            logger.info(f"Attempting to load ViT processor for model ID: {self.model_id}")
            self.processor = AutoImageProcessor.from_pretrained(self.model_id, use_fast=True)
            logger.info("ViT processor loaded successfully.")
            processor_loaded = True
        except Exception as e:
            # Log at error level as processor is important but not strictly critical if we raise later
            logger.error(f"An error occurred while loading the ViT processor for model ID {self.model_id}: {e}", exc_info=True)
            # Do not re-raise here, continue loading other artifacts


        # Load Model
        try:
            logger.info(f"Attempting to load ViT model from {model_path}")
            # Note: Adjust map_location as needed based on where the model was saved
            self.model = torch.load(model_path, map_location=self.device, weights_only=False)
            self.model.to(self.device) # Ensure model is on the correct device
            logger.info(f"ViT model loaded successfully and moved to {self.device}.")
            model_loaded = True
        except FileNotFoundError:
            logger.critical(f"Critical Error: ViT model file not found at {model_path}", exc_info=True)
            raise # Re-raise to indicate a critical initialization failure
        except Exception as e:
            logger.critical(f"Critical Error: An unexpected error occurred while loading the ViT model from {model_path}: {e}", exc_info=True)
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

        if processor_loaded and model_loaded and mlb_loaded:
             logger.info("All required ViT artifacts loaded successfully.")
        else:
            logger.error("One or more required ViT artifacts failed to load during _load_artifacts.")


    def predict(self, image_path: Path, cut_off: float = 0.5) -> Optional[List[str]]:
        """
        Predicts the class labels for a given image using the loaded ViT model.

        The process involves loading and preprocessing the image, performing
        inference with the model, applying a sigmoid activation, thresholding
        the probabilities to obtain binary predictions, and finally converting
        the binary predictions back to class labels using the MultiLabelBinarizer.

        Args:
            image_path: Path to the image file to classify. The image is expected
                        to be in a format compatible with PIL (Pillow).
            cut_off: The threshold for converting predicted probabilities into
                     binary labels. Probabilities greater than or equal to this
                     value are considered positive predictions (1), otherwise 0.
                     Defaults to 0.5.

        Returns:
            A list of predicted class labels (strings) if the prediction process
            is successful. Returns None if any critical step (image loading,
            preprocessing, model inference, or inverse transform) fails.
            Returns an empty list if the prediction process is successful but
            no labels meet the cutoff threshold.
        """
        logger.info(f"Starting prediction process for image: {image_path} with cutoff {cut_off}.")

        if self.model is None or self.processor is None or self.mlb is None:
            logger.error("Model, processor, or MultiLabelBinarizer not loaded. Cannot perform prediction.")
            return None

        # Load and preprocess image
        image: Optional[Image.Image] = None
        try:
            logger.info(f"Attempting to load image from {image_path}")
            image = Image.open(image_path)
            logger.info(f"Image loaded successfully from {image_path}.")
        except FileNotFoundError:
            logger.error(f"Error: Image file not found at {image_path}", exc_info=True)
            return None
        except Exception as e:
            logger.error(f"An unexpected error occurred while loading image {image_path}: {e}", exc_info=True)
            return None

        try:
            logger.info(f"Attempting to convert image to RGB for {image_path}.")
            if image.mode != "RGB":
                image = image.convert("RGB")
                logger.info(f"Image converted to RGB successfully for {image_path}.")
            else:
                 logger.info(f"Image is already in RGB format for {image_path}.")

        except Exception as e:
            logger.error(f"An error occurred while converting image {image_path} to RGB: {e}", exc_info=True)
            return None


        # Preprocess image using the loaded processor
        try:
            logger.info(f"Attempting to preprocess image using processor for {image_path}.")
            # Check if image is valid after loading/conversion
            if image is None:
                 logger.error(f"Image is None after loading/conversion for {image_path}. Cannot preprocess.")
                 return None
            # The processor expects a PIL Image or a list of PIL Images
            pixel_values: torch.Tensor = self.processor(images=image, return_tensors="pt").pixel_values.to(self.device)
            logger.info(f"Image preprocessed and moved to device ({self.device}).")
        except Exception as e:
            logger.error(f"An error occurred during image preprocessing for {image_path}: {e}", exc_info=True)
            return None

        # Forward pass
        try:
            logger.info(f"Starting model forward pass for {image_path}.")
            self.model.eval() # Set model to evaluation mode
            with torch.no_grad():
                outputs: Any = self.model(pixel_values) # Use Any because the output type can vary
                logits: torch.Tensor = outputs.logits
            logger.info(f"Model forward pass completed for {image_path}.")
        except Exception as e:
            logger.error(f"An error occurred during model forward pass for {image_path}: {e}", exc_info=True)
            return None


        # Apply sigmoid and thresholding
        try:
            logger.info(f"Applying sigmoid and thresholding for {image_path}.")
            sigmoid: torch.nn.Sigmoid = torch.nn.Sigmoid()
            probs: torch.Tensor = sigmoid(logits.squeeze().cpu())

            predictions: np.ndarray = np.zeros(probs.shape, dtype=int) # Explicitly set dtype to int
            print(predictions)
            predictions[np.where(probs >= cut_off)] = 1
            logger.info(f"Applied sigmoid and thresholding with cutoff {cut_off} for {image_path}. Binary predictions shape: {predictions.shape}")
        except Exception as e:
            logger.error(f"An error occurred during probability processing for {image_path}: {e}", exc_info=True)
            return None


        # Get label names using the loaded MultiLabelBinarizer
        try:
            logger.info(f"Performing inverse transform using MultiLabelBinarizer for {image_path}.")
            # The predictions need to be in a 2D array for inverse_transform, e.g., (1, num_classes)
            # Use the self.mlb loaded during initialization

            # Ensure self.mlb is not None (checked at the start of predict, but good practice)
            if self.mlb is None:
                 logger.error(f"MultiLabelBinarizer is None. Cannot perform inverse transform for {image_path}.")
                 return None

            binary_prediction: np.ndarray

            # Ensure predictions shape is compatible (must be 2D: (n_samples, n_classes))
            # Since we process one image at a time, expected shape is (1, n_classes)
            expected_shape: Tuple[int, int] = (1, len(self.mlb.classes_))

            if predictions.ndim == 1 and predictions.shape[0] == len(self.mlb.classes_):
                 binary_prediction = predictions.reshape(expected_shape)
                 logger.info(f"Reshaped 1D prediction to 2D ({expected_shape}) for inverse transform.")
            elif predictions.ndim == 2 and predictions.shape == expected_shape:
                 binary_prediction = predictions
                 logger.info(f"Prediction already in correct 2D shape ({expected_shape}) for inverse transform.")
            else:
                 logger.error(f"Cannot inverse transform prediction shape {predictions.shape} with MLB classes {len(self.mlb.classes_)} for {image_path}. Expected shape: {expected_shape}")
                 return None


            predicted_labels_tuple_list: List[Tuple[str, ...]] = self.mlb.inverse_transform(binary_prediction)
            logger.info(f"Prediction processed for {image_path}. Predicted labels (raw tuple list): {predicted_labels_tuple_list}")

            # inverse_transform returns a list of tuples, even for a single sample.
            # We expect a single prediction here, so we take the first tuple.
            if predicted_labels_tuple_list and len(predicted_labels_tuple_list) > 0:
                final_labels: List[str] = list(predicted_labels_tuple_list[0])
                logger.info(f"Final predicted labels for {image_path}: {final_labels}")
                return final_labels
            else:
                 logger.warning(f"MLB inverse_transform returned an empty list for {image_path}. No labels predicted.")
                 return []


        except Exception as e:
            logger.error(f"An error occurred during inverse transform for {image_path}: {e}", exc_info=True)
            return None
        

