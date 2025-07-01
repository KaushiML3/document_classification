from documentClassifire.src.steps import *
from zenml.pipelines import pipeline
from documentClassifire.configuration.cofiguration_setup import ConfigManager
from zenml.pipelines import pipeline



@pipeline(enable_cache=True)
def training_pipeline():
    # Configuration
    config = config_management_step()
    
    # Data handling
    data_split= data_ingestion_step(config)
    train_paths = data_split[0]
    val_paths = data_split[1]
    test_paths = data_split[2]

    custom_datagen = preprocess_step(config)
    
    # MLflow setup
    mlflow_connection_step(config)
    
    # Model building
    cnn_model = cnn_model_build_step(config)
    vgg_model = vgg16_model_build_step(config)
    eff_model = efficient_net_b0_model_build_step(config)

    # Model config 
    cnn_trainer_config = get_cnn_model_trainer_config_step(config)
    vgg_trainer_config = get_vgg_model_trainer_config_step(config)
    eff_trainer_config = get_eff_model_trainer_config_step(config)

    # Model training
    train_model_step(
        model=cnn_model,
        trainer_config=cnn_trainer_config,
        datagen=custom_datagen,
        train_paths=train_paths,
        val_paths=val_paths,
        test_paths=test_paths
    )
    
    train_model_step(
        model=vgg_model,
        trainer_config=vgg_trainer_config,
        datagen=custom_datagen,
        train_paths=train_paths,
        val_paths=val_paths,
        test_paths=test_paths
    )
    
    train_model_step(
        model=eff_model,
        trainer_config=eff_trainer_config,
        datagen=custom_datagen,
        train_paths=train_paths,
        val_paths=val_paths,
        test_paths=test_paths
    )