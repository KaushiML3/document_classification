# Document_classification

This project implements three main techniques for document and image classification:

## 1. Image Classification
- **Models:** VGG16, EfficientNet
- **Description:** Standard image classification to assign a single label to each image.
- [`notebook/prediction.ipynb`](notebook/prediction.ipynb)

## 2. Image Multilabel Classification
- **Models:** Vision Transformer (ViT), Custom Architecture
- **Description:** Assigns single labels to each image using advanced architectures.
- [`notebook/prediction.ipynb`](notebook/prediction.ipynb)

## 3. Document Layout Classification
- **Model:** LayoutLMv2
- **Description:** Classifies documents based on their layout and structure using LayoutLMv2.
- [`notebook/prediction.ipynb`](notebook/prediction.ipynb)

## Dataset

- **Location:** (link)[https://www.kaggle.com/datasets/kaushigihanml/text-document-images]
- **Structure:** The dataset is organized into `train` and `test` folders, each containing the following document classes:
  - `note`
  - `invoice`
  - `form`
  - `email`
  - `advertisement`
- **Format:** Each class folder contains multiple `.jpg` image files representing scanned or photographed documents of that type.
- **Usage:** The dataset is used for both image and document layout classification tasks, supporting single-label and multi-label experiments.

---

## Experiment Tracking & Model Registry
- **MLflow & DagsHub:** Used for tracking experiments and managing model registry throughout the project lifecycle.
    <img src="path/to/vgg16_confusion_matrix.png" alt="Mlflow" width="400"/>
    <img src="path/to/vgg16_confusion_matrix.png" alt="Mlflow" width="400"/>
    <img src="path/to/vgg16_confusion_matrix.png" alt="dagshub" width="400"/>

## Workflow Management
- **ZenML:** Utilized as the workflow manager to orchestrate and automate the machine learning pipelines.
    <img src="path/to/vgg16_confusion_matrix.png" alt="Mlflow" width="400"/>

## End-to-End Pipelines
- Developed complete end-to-end pipelines for training CNN, VGG, and EfficientNet models, covering data ingestion, preprocessing, model training, and evaluation.

  **Pipeline Steps:**
  1. Configuration Management: Load and manage all configuration settings.
  2. Data Ingestion: Collect and split image data into training, validation, and test sets.
  3. Preprocessing: Prepare data generators for model training.
  4. MLflow Connection: Set up experiment tracking and model registry.
  5. Model Building: Build the desired model (CNN, VGG16, or EfficientNetB0).
  6. Trainer Configuration: Load the appropriate training configuration for the selected model.
  7. Model Training: Train the model using the prepared data and configuration.

## How to Run the Pipeline

1. **Install dependencies:**
   ```bash
   pip install -r requirements.txt
   ```
2. **Start the ZenML server:**
   ```bash
   zenml up
   ```
3. **Run the training pipeline:**
   ```bash
   python trainer.py
   ```

## API & Demo

- **FastAPI Endpoint:**  
  A FastAPI server is implemented to provide RESTful API endpoints for model inference and integration.
    <img src="path/to/vgg16_confusion_matrix.png" alt="Mlflow" width="400"/>

- **Hugging Face Gradio Application:**  
  An interactive Gradio web app is available for live model demos and testing.
    <img src="path/to/vgg16_confusion_matrix.png" alt="Mlflow" width="400"/>

- **Try it on Hugging Face Spaces:**  
  [![Gradio App](https://img.shields.io/badge/Gradio-Demo-blue?logo=gradio)](https://huggingface.co/spaces/your-username/your-space-name)

## Challenges

- **MLflow Model Deployment:** Faced issues with model deployment due to version incompatibilities between MLflow and other dependencies.
- **Detectron Installation on Windows:** Encountered difficulties installing Detectron on Windows, which is required to run LayoutLM models on the local machine.

## Future Improvements

- **Larger Dataset:** Plan to create and curate a larger, more diverse dataset to improve model performance and generalization.
- **Advanced Experiments:** Intend to conduct further experiments with LayoutLM and ViT models, especially for document layout analysis and classification tasks.



## Model Confusion Matrix Comparison

Add the confusion matrix images for each model below:

- **VGG16:**
  <img src="path/to/vgg16_confusion_matrix.png" alt="VGG16 Confusion Matrix" width="400"/>

- **EfficientNet:**
  <img src="path/to/efficientnet_confusion_matrix.png" alt="EfficientNet Confusion Matrix" width="400"/>

- **ViT:**
  <img src="path/to/vit_confusion_matrix.png" alt="ViT Confusion Matrix" width="400"/>

- **Custom Architecture:**
  <img src="path/to/custom_confusion_matrix.png" alt="Custom Architecture Confusion Matrix" width="400"/>

- **LayoutLMv2:**
  <img src="path/to/layoutlmv2_confusion_matrix.png" alt="LayoutLMv2 Confusion Matrix" width="400"/>





