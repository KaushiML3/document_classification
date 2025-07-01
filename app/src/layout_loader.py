from PIL import Image
import numpy as np
import torch
from typing import Optional, List, Dict, Any
from pathlib import Path
from transformers import LayoutLMv2ForSequenceClassification, LayoutLMv2Processor, LayoutLMv2FeatureExtractor, LayoutLMv2Tokenizer
import os
from dotenv import load_dotenv
from app.src.logger import setup_logger



logger = setup_logger("layout_loader")

class LayoutLMDocumentClassifier:
    """
    A class for classifying documents using a LayoutLMv2 model.

    This class encapsulates the loading of the LayoutLMv2 model and its associated
    processor, handles image preprocessing, and performs document classification
    predictions. The model path is loaded from environment variables, promoting
    flexible configuration. It includes robust error handling, logging, and
    type hinting for production readiness.
    """

    def __init__(self,model_path_str) -> None:
        """
        Initializes the LayoutLMDocumentClassifier by loading the model and processor.

        The model and processor are loaded from the path specified in the
        environment variable 'LAYOUTLM_MODEL_PATH'. This method also sets up
        the device for inference (GPU if available, otherwise CPU) and defines
         the mapping from model output indices to class labels.

        Includes robust error handling and logging for initialization and artifact loading.

        Raises:
            ValueError: If the 'LAYOUTLM_MODEL_PATH' environment variable is not set.
            FileNotFoundError: If the model path specified in the environment variable
                               does not exist or a required artifact file is not found
                               during the artifact loading process.
            Exception: If any other unexpected error occurs during the loading
                       of the model or processor.
        """
        logger.info("Initializing LayoutLMDocumentClassifier.")
        self.model_path_str: Optional[str]=model_path_str
        self.model: Optional[LayoutLMv2ForSequenceClassification] = None
        self.processor: Optional[LayoutLMv2Processor] = None
        self.device: torch.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        logger.info(f"Using device: {self.device}")
        # Define id2label mapping as a class attribute
        # This mapping should align with the model's output classes.
        self.id2label: Dict[int, str] = {0:'invoice', 1: 'form', 2:'note', 3:'advertisement', 4: 'email'}
        logger.info(f"Defined id2label mapping: {self.id2label}")

        # Load model path from environment variable
        model_path_str: Optional[str] = self.model_path_str
        logger.info(f"Attempting to retrieve LAYOUTLM_MODEL_PATH from environment variables.")
        if not model_path_str:
            logger.critical("Critical Error: 'LAYOUTLM_MODEL_PATH' environment variable is not set.")
            raise ValueError("LAYOUTLM_MODEL_PATH environment variable is not set.")

        model_path: Path = Path(model_path_str)
        logger.info(f"Retrieved model path: {model_path}")
        if not model_path.exists():
             logger.critical(f"Critical Error: Model path from environment variable does not exist: {model_path}")
             raise FileNotFoundError(f"Model path not found: {model_path}")
        logger.info(f"Model path {model_path} exists.")


        try:
            logger.info("Calling _load_artifacts to load model and processor.")
            self._load_artifacts(model_path)
            if self.model is not None and self.processor is not None:
                logger.info("LayoutLMDocumentClassifier initialized successfully.")
            else:
                # This case should ideally be caught and re-raised in _load_artifacts
                logger.critical("LayoutLMDocumentClassifier failed to fully initialize due to artifact loading errors in _load_artifacts.")
                # _load_artifacts already raises on critical failure, no need to raise again
        except Exception as e:
            # Catch and log any exception that wasn't handled and re-raised in _load_artifacts
            logger.critical(f"An unhandled exception occurred during LayoutLMDocumentClassifier initialization: {e}", exc_info=True)
            raise # Re-raise the exception after logging
        logger.info("Initialization process completed.")


    def _load_artifacts(self, model_path: Path) -> None:
        """
        Loads the LayoutLMv2 model and processor from the specified path.

        This is an internal helper method called during initialization. It handles
        the loading of both the `LayoutLMv2ForSequenceClassification` model and
        its corresponding `LayoutLMv2Processor` with error handling and logging.

        Args:
            model_path: Path to the LayoutLMv2 model directory or file. This path
                        is expected to contain both the model weights and the
                        processor configuration/files.

        Raises:
            FileNotFoundError: If the `model_path` or any required processor/model
                               file within that path is not found.
            Exception: If any other unexpected error occurs during loading
                       from the specified path (e.g., corrupt files, compatibility issues).
        """
        logger.info(f"Starting artifact loading from {model_path} for LayoutLMv2.")
        processor_loaded: bool = False
        model_loaded: bool = False

        # Load Processor
        try:
            logger.info(f"Attempting to load LayoutLMv2 processor from {model_path}")
            # Load feature extractor and tokenizer separately to create the processor
            feature_extractor = LayoutLMv2FeatureExtractor()
            tokenizer = LayoutLMv2Tokenizer.from_pretrained("microsoft/layoutlmv2-base-uncased")
            self.processor = LayoutLMv2Processor(feature_extractor, tokenizer)
            logger.info("LayoutLMv2 processor loaded successfully.")
            processor_loaded = True
        except Exception as e:
            logger.critical(f"Critical Error: An unexpected error occurred while loading the LayoutLMv2 processor from {model_path}: {e}", exc_info=True)
            raise # Re-raise to indicate a critical initialization failure

        # Load Model
        try:
            logger.info(f"Attempting to load LayoutLMv2 model from {model_path}")
            self.model = LayoutLMv2ForSequenceClassification.from_pretrained(model_path)
            self.model.to(self.device) # Ensure model is on the correct device
            logger.info(f"LayoutLMv2 model loaded successfully and moved to {self.device}.")
            model_loaded = True
        except FileNotFoundError:
            logger.critical(f"Critical Error: LayoutLMv2 model file not found at {model_path}", exc_info=True)
            raise # Re-raise to indicate a critical initialization failure
        except Exception as e:
            logger.critical(f"Critical Error: An unexpected error occurred while loading the LayoutLMv2 model from {model_path}: {e}", exc_info=True)
            raise # Re-raise to indicate a critical initialization failure

        # Conditional logging based on loading success
        if model_loaded and processor_loaded:
             logger.info("All required LayoutLMv2 artifacts loaded successfully from _load_artifacts.")
        elif model_loaded and not processor_loaded:
             logger.error("LayoutLMv2 model loaded successfully, but processor loading failed in _load_artifacts.")
        elif not model_loaded and processor_loaded:
             logger.error("LayoutLMv2 processor loaded successfully, but model loading failed in _load_artifacts.")
        else:
            logger.error("Both LayoutLMv2 model and processor failed to load during _load_artifacts.")
        logger.info("Artifact loading process completed.")


    def _prepare_inputs(self, image_path: Path) -> Optional[Dict[str, torch.Tensor]]:
        """
        Loads and preprocesses an image to prepare inputs for the LayoutLMv2 model.

        This method handles loading the image from a file path, converting it to RGB,
        and using the loaded LayoutLMv2Processor to create the necessary input tensors
        (pixel values, input IDs, attention masks, bounding boxes). The tensors are
        then moved to the appropriate device for inference.

        Includes robust error handling and logging for each step.

        Args:
            image_path: Path to the image file (e.g., PNG, JPG) to be processed.

        Returns:
            A dictionary containing the prepared input tensors (e.g., 'pixel_values',
            'input_ids', 'attention_mask', 'bbox') as PyTorch tensors, if image
            loading and preprocessing are successful. Returns `None` if any
            step fails (e.g., file not found, image corruption, processor error).
        """
        logger.info(f"Starting image loading and preprocessing for {image_path}.")
        image: Optional[Image.Image] = None

        # Load image
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

        # Convert image to RGB
        try:
            logger.info(f"Attempting to convert image to RGB for {image_path}.")
            if image is None:
                 logger.error(f"Image is None after loading for {image_path}. Cannot convert to RGB.")
                 return None
            if image.mode != "RGB":
                image = image.convert("RGB")
                logger.info(f"Image converted to RGB successfully for {image_path}.")
            else:
                 logger.info(f"Image is already in RGB format for {image_path}.")

        except Exception as e:
            logger.error(f"An error occurred while converting image {image_path} to RGB: {e}", exc_info=True)
            return None


        # Prepare inputs using the processor
        if self.processor is None:
            logger.error("LayoutLMv2 processor is not loaded. Cannot prepare inputs.")
            return None

        encoded_inputs: Optional[Dict[str, torch.Tensor]] = None
        try:
            logger.info(f"Attempting to prepare inputs using processor for {image_path}.")
            # The processor expects a PIL Image or a list of PIL Images
            if image is None:
                 logger.error(f"Image is None before preprocessing for {image_path}. Cannot prepare inputs.")
                 return None

            encoded_inputs = self.processor(
                images=image,
                return_tensors="pt",
                truncation=True,
                padding="max_length",
                max_length=512
            )
            logger.info(f"Inputs prepared successfully for {image_path}.")
        except Exception as e:
            logger.error(f"An error occurred during input preparation for {image_path}: {e}", exc_info=True)
            return None

        # Move inputs to the device
        if encoded_inputs is not None:
            try:
                logger.info(f"Attempting to move inputs to device ({self.device}) for {image_path}.")
                for k, v in encoded_inputs.items():
                    if isinstance(v, torch.Tensor):
                        encoded_inputs[k] = v.to(self.device)
                logger.info(f"Inputs moved to device ({self.device}) successfully for {image_path}.")
            except Exception as e:
                logger.error(f"An error occurred while moving inputs to device for {image_path}: {e}", exc_info=True)
                return None
        else:
             logger.error(f"Encoded inputs are None after processing for {image_path}. Cannot move to device.")
             return None


        logger.info(f"Image loading and preprocessing completed successfully for {image_path}.")
        return encoded_inputs


    def predict(self, image_path: Path) -> Optional[str]:
        """
        Predicts the class label for a given image using the loaded LayoutLMv2 model.

        This is the main prediction method. It orchestrates the process by first
        preparing the image inputs using `_prepare_inputs`, performing inference
        with the LayoutLMv2 model, determining the predicted class index from the
        model's output logits, and finally mapping this index to a human-readable
        class label using the `id2label` mapping.

        Includes robust error handling and logging throughout the prediction pipeline.

        Args:
            image_path: Path to the image file to classify.

        Returns:
            The predicted class label as a string if the entire prediction process
            is successful. Returns `None` if any critical step fails (e.g.,
            image loading/preprocessing, model inference, or if the predicted
            index is not found in the `id2label` mapping).
        """
        logger.info(f"Starting prediction process for image: {image_path}.")

        # Prepare inputs
        logger.info(f"Calling _prepare_inputs for {image_path}.")
        encoded_inputs: Optional[Dict[str, torch.Tensor]] = self._prepare_inputs(image_path)
        if encoded_inputs is None:
            logger.error(f"Input preparation failed for {image_path}. Cannot perform prediction.")
            logger.info(f"Prediction process failed for {image_path}.")
            return None
        logger.info(f"Input preparation successful for {image_path}.")


        # Check if model is loaded
        if self.model is None:
            logger.error("LayoutLMv2 model is not loaded. Cannot perform prediction.")
            logger.info(f"Prediction process failed for {image_path}.")
            return None
        logger.info("LayoutLMv2 model is loaded. Proceeding with inference.")

        predicted_label: Optional[str] = None

        try:
            logger.info(f"Performing model inference for {image_path}.")
            self.model.eval() # Set model to evaluation mode
            with torch.no_grad():
                outputs: Any = self.model(**encoded_inputs)
                logits: torch.Tensor = outputs.logits

            # Determine predicted class index
            # Ensure logits is a tensor before calling argmax
            if not isinstance(logits, torch.Tensor):
                 logger.error(f"Model output 'logits' is not a torch.Tensor for {image_path}. Cannot determine predicted index.")
                 logger.info(f"Prediction process failed for {image_path} due to invalid model output.")
                 return None

            predicted_class_idx: int = logits.argmax(-1).item()
            logger.info(f"Model inference completed for {image_path}. Predicted index: {predicted_class_idx}.")

            # Map index to label
            logger.info(f"Attempting to map predicted index {predicted_class_idx} to label.")
            if predicted_class_idx in self.id2label:
                predicted_label = self.id2label[predicted_class_idx]
                logger.info(f"Mapped predicted index {predicted_class_idx} to label: {predicted_label}.")
            else:
                logger.error(f"Predicted index {predicted_class_idx} not found in id2label mapping for {image_path}.")
                logger.info(f"Prediction process failed for {image_path} due to unknown predicted index.")
                return None # Return None if index is not in mapping

        except Exception as e:
            logger.error(f"An error occurred during model inference or label mapping for {image_path}: {e}", exc_info=True)
            logger.info(f"Prediction process failed for {image_path} due to inference/mapping error.")
            return None

        logger.info(f"Prediction process completed successfully for {image_path}. Predicted label: {predicted_label}.")
        return predicted_label
    

