from tensorflow.keras.optimizers import Adam
from mlflow.models import infer_signature
import keras
import os
import mlflow
import mlflow.tensorflow
from tensorflow.keras.callbacks import ModelCheckpoint
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix, classification_report
from typing import List
from documentClassifire.utils.utility import save_model_to_artifacts, save_joblib
from documentClassifire.entity.config_entity import ModelTrainerConfig
from tensorflow.keras.utils import plot_model
from documentClassifire.logger.logger import setup_logger



from documentClassifire.src.preprocess import *
from pathlib import Path
import numpy as np
import tensorflow as tf

logger = setup_logger()


class ModelTrainer():
    def __init__(self,train_paths:List, val_paths:List, test_paths:List):
      self.train_paths=train_paths
      self.val_paths=val_paths
      self.test_paths=test_paths


    @staticmethod
    def log_training_plots(directory,history, run_id):
        """Log training history plots to MLflow."""

        # Plot training history
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4))

        ax1.plot(history.history["loss"], label="Training Loss")
        ax1.plot(history.history["val_loss"], label="Validation Loss")
        ax1.set_title("Model Loss")
        ax1.set_xlabel("Epoch")
        ax1.set_ylabel("Loss")
        ax1.legend()

        ax2.plot(history.history["accuracy"], label="Training Accuracy")
        ax2.plot(history.history["val_accuracy"], label="Validation Accuracy")
        ax2.set_title("Model Accuracy")
        ax2.set_xlabel("Epoch")
        ax2.set_ylabel("Accuracy")
        ax2.legend()

        plt.tight_layout()
        path=os.path.join(directory,"training_history.png")
        plt.savefig(path, dpi=300, bbox_inches="tight")
        mlflow.log_artifact(path)
        plt.close()

    @staticmethod
    def log_evaluation_metrics_classification(directory,model,dataset):
        """Log comprehensive evaluation metrics."""

        all_true_labels = []
        all_pred_labels = []

        label2id = dataset.label2id
        id2label = dataset.id2label
        class_names = [id2label[i] for i in range(len(id2label))]

        # Get predictions
        for i in range(len(dataset)):
            images, true_labels = dataset[i]  # batched data

            preds = model.predict(images)  # shape: (batch_size, num_classes)

            # For multi-class (single-label) classification: use argmax
            pred_ids = np.argmax(preds, axis=1)
            true_ids = np.argmax(true_labels, axis=1)

            all_true_labels.extend(true_ids)
            all_pred_labels.extend(pred_ids)

        # Confusion matrix
        cm = confusion_matrix(all_true_labels, all_pred_labels)
        plt.figure(figsize=(10, 8))
        sns.heatmap(
            cm,
            annot=True,
            fmt="d",
            cmap="Blues",
            xticklabels=class_names,
            yticklabels=class_names,
        )
        plt.title("Confusion Matrix")
        plt.ylabel("True Label")
        plt.xlabel("Predicted Label")
        path=os.path.join(directory,"confusion_matrix.png")
        plt.savefig(path, dpi=300, bbox_inches="tight")
        mlflow.log_artifact(path)
        plt.close()

        # Classification report
        report = classification_report(
            all_true_labels, all_pred_labels, target_names=class_names, output_dict=True,zero_division=0
        )

        # Log per-class metrics
        for class_name in class_names:
            if class_name in report:
                mlflow.log_metrics(
                    {
                        f"{class_name}_precision": report[class_name]["precision"],
                        f"{class_name}_recall": report[class_name]["recall"],
                        f"{class_name}_f1": report[class_name]["f1-score"],
                    }
                )

    
         






    def model_trainer(self,model,model_trainer_config:ModelTrainerConfig,custom_datagen):
        # You can use 'tensorflow', 'torch', or 'jax' as backend
        # Make sure to set the environment variable before importing Keras
        os.environ["KERAS_BACKEND"] = "tensorflow"


        # Enable autologging for TensorFlow/Keras
        mlflow.tensorflow.autolog()

        experiment_name=model_trainer_config.experiment_name
        run_name=model_trainer_config.run_name
        params = {
        "epochs": model_trainer_config.epochs,
        "batch_size":model_trainer_config.batch_size,
        "learning_rate":model_trainer_config.learning_rate,
        "optimizer":model_trainer_config.optimizer,
        "loss_function":model_trainer_config.loss_function,
        "dropout_rate":model_trainer_config.dropout_rate,
        "hidden_units":model_trainer_config.hidden_units,
        "input_shape":model_trainer_config.input_shape
        }

        experiment = mlflow.get_experiment_by_name(experiment_name)
        if experiment is None:
            experiment_id = mlflow.create_experiment(experiment_name)
        else:
            experiment_id = experiment.experiment_id


        with mlflow.start_run(experiment_id=experiment_id, run_name=run_name, nested=True):
        #with mlflow.start_run(nested=True):

            
            logger.info("Log params")
            # Log training parameters
            mlflow.log_params(params)

            logger.info("Compile the model")
            # Create and compile model
            loss=params["loss_function"]
            if loss == "categorical_crossentropy":
                model.compile(
                    optimizer=keras.optimizers.Adam(learning_rate=params["learning_rate"]),
                    loss=keras.losses.CategoricalCrossentropy(),
                    metrics=["accuracy"],
                )

            elif loss=="binary_crossentropy":
                model.compile(
                    optimizer=keras.optimizers.Adam(learning_rate=params["learning_rate"]),
                    loss=keras.losses.BinaryCrossentropy(),
                    metrics=["accuracy"],
                )

            # get the datasets
            logger.info("Start to preprocess the data ")
            train_dataset = Dataset(self.train_paths, batch_size=params["batch_size"], input_shape=params["input_shape"],datagen=custom_datagen, augment=True)
            val_dataset = Dataset(self.val_paths, batch_size=params["batch_size"], input_shape=params["input_shape"],datagen=custom_datagen, augment=False)
            test_dataset = Dataset(self.test_paths,batch_size=params["batch_size"], input_shape=params["input_shape"],datagen=custom_datagen, augment=False)


            directory=save_model_to_artifacts(experiment_id=Path(experiment_name))


            logger.info("save model summery")
            # Log model architecture
            model_summery_path=os.path.join(directory,"model_summary.txt")
            with open(model_summery_path, "w", encoding="utf-8") as f:
                model.summary(print_fn=lambda x: f.write(x + "\n"))
            mlflow.log_artifact(model_summery_path)



            logger.info("save model architecture")
            # Log model visualization
            try:
                model_architecture_path=os.path.join(directory,"model_architecture.png")
                plot_model(model, to_file=model_architecture_path, show_shapes=True)
                mlflow.log_artifact(model_architecture_path)
            except Exception as e:
                logger.error("Model plot not generated: %s", e)

            print("Define the callback")
            # Define the ModelCheckpoint callback

            model_path=os.path.join(directory, "model.keras")
            checkpoint = ModelCheckpoint(model_path, save_best_only=True, monitor='val_loss', verbose=1)
            #model.save(model_path)
            ## save
            multilabel_binanzer=train_dataset.mlb
            mlb_path=os.path.join(directory, "mlb.joblib")
            save_joblib(multilabel_binanzer, mlb_path)


            # Custom callback for logging metrics
            class MLflowCallback(keras.callbacks.Callback):
                def on_epoch_end(self, epoch, logs=None):
                    if logs:
                        mlflow.log_metrics(
                            {
                                "train_loss": logs.get("loss"),
                                "train_accuracy": logs.get("accuracy"),
                                "val_loss": logs.get("val_loss"),
                                "val_accuracy": logs.get("val_accuracy"),
                            },
                            step=epoch,
                        )
            # Prepare sample data for signature inference
            sample_input = train_dataset[0][0]
            sample_predictions = model.predict(sample_input)

            # Infer signature from sample data
            signature = infer_signature(sample_input, sample_predictions)

            # Train model with custom callback
            history = model.fit(
                train_dataset,
                batch_size=params["batch_size"],
                epochs=params["epochs"],
                validation_data=val_dataset,
                callbacks=[MLflowCallback(),checkpoint],
                verbose=1,
            )

            # Evaluate model
            #ModelTrainer.log_training_plots(directory,history, mlflow.active_run().info.run_id)
            ModelTrainer.log_evaluation_metrics_classification(directory,model,test_dataset)



            #mlflow.keras.log_model(model, name="model",signature=signature)
            #mlflow.keras.log_model(
            #    model,
            #   name="keras_model",
            #   signature=signature,
            #    input_example=sample_input,
             #   registered_model_name="MyKerasHandwrittenDigitRecognizer" # This registers the model
            #)

            if os.path.exists(model_path) and os.path.exists(mlb_path):
              mlflow.log_artifact(model_path)
              logger.info("Model saved in run %s", mlflow.active_run().info.run_id)
              mlflow.log_artifact(mlb_path)
              logger.info("Model saved in run %s", mlflow.active_run().info.run_id)

            else:
              logger.error("model not save")
