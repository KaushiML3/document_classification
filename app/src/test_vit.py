import joblib
from sklearn.preprocessing import MultiLabelBinarizer
from pathlib import Path
import torch
import numpy as np
from PIL import Image
from transformers import AutoImageProcessor, AutoModelForImageClassification
from app.src.logger import setup_logger

logger = setup_logger("test_vit")

try:
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    mlb_file_path=Path("artifacts\model\VIT_model\mlb.joblib")
    model_file_path=Path("artifacts\model\VIT_model\model.pth")
    # Select model
    model_id = "google/vit-base-patch16-224-in21k"
    # Load processor
    processor = AutoImageProcessor.from_pretrained(model_id, use_fast=True)

    # TODO: You need to load your fine-tuned model here
    # For example:
    # model = AutoModelForImageClassification.from_pretrained("path/to/your/fine-tuned-model")
    # For now, we will use the base model for demonstration, but it will not give correct predictions.
    #model = AutoModelForImageClassification.from_pretrained(model_id)
    # Load the entire model
    model= torch.load(model_file_path, map_location=device,weights_only=False )
    # Set device
    model.to(device)

except Exception as e:
    logger.error(str(e))
    raise e




def mlb_load(file_path:Path)->MultiLabelBinarizer:
    try:
        # Assuming you run this notebook from the root of your project directory
        mlb = joblib.load(file_path)

    except FileNotFoundError:
        logger.error("Error: 'artifacts/model/VIT_model/mlb.joblib' not found.")
        logger.error("Please make sure the path is correct. Using a placeholder binarizer.")
        # As a placeholder, let's create a dummy mlb if the file is not found.
        mlb = MultiLabelBinarizer()
        # This should be the set of your actual labels.
        mlb.fit([['advertisement', 'email', 'form', 'invoice', 'note']])
    return mlb






def VIT_model_prediction(image_path:Path,cut_off:float):
    try:
        # Load and convert image
        # --- IMPORTANT: Please update this path to your image ---
        try:
            image = Image.open(image_path)
            if image.mode != "RGB":
                image = image.convert("RGB")
        except FileNotFoundError:
            logger.error(f"Error: Image not found at {image_path}")
            logger.error("Using a dummy image for demonstration.")
            # Create a dummy image for demonstration if image not found
            image = Image.new('RGB', (224, 224), color = 'red')


        # Preprocess image
        pixel_values = processor(image, return_tensors="pt").pixel_values.to(device)

        # Forward pass
        with torch.no_grad():
            outputs = model(pixel_values)
            logits = outputs.logits

        # Apply sigmoid for multi-label classification
        sigmoid = torch.nn.Sigmoid()
        probs = sigmoid(logits.squeeze().cpu())

        # Thresholding (using 0.5 as an example)
        predictions = np.zeros(probs.shape)
        predictions[np.where(probs >= cut_off)] = 1

        # Get label names using the loaded MultiLabelBinarizer
        mlb=mlb_load(mlb_file_path)
        # The predictions need to be in a 2D array for inverse_transform, e.g., (1, num_classes)
        predicted_labels = mlb.inverse_transform(predictions.reshape(1, -1))
        logger.info(f"Predicted labels: {predicted_labels}")
        return {"status":1,"classe":predicted_labels}

    except Exception as e:
        logger.error(str(e))
        raise e



#VIT_model_prediction(Path(r"dataset\sample_text_ds\test\email\2078379610a.jpg"),0.5)