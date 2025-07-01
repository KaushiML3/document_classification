import yaml
from box import ConfigBox
from ensure import ensure_annotations
from box.exceptions import BoxValueError
from pathlib import Path
import yaml
import os
import pickle
import joblib
import os
from datetime import datetime
from pathlib import Path

@ensure_annotations
def read_yaml_file(file_path:Path):

    """reads yaml file and returns

    Args:
        path_to_yaml (str): path like input

    Raises:
        ValueError: if yaml file is empty
        e: empty file

    Returns:
        ConfigBox: ConfigBox type
    """
    with open(file_path, 'r') as file:
        try:
            yaml_data = yaml.safe_load(file)
            return ConfigBox(yaml_data)
        except yaml.YAMLError as e:
            print(f"Error reading YAML file {file_path}: {e}")



@ensure_annotations
def write_yaml_file(data:dict, file_path:Path):
    """

    """
    with open(file_path, 'w') as file:
        try:
            yaml.dump(data, file, default_flow_style=False)
            print(f"Data successfully written to {file_path}")
        except yaml.YAMLError as e:
            print(f"Error writing YAML file {file_path}: {e}")


@ensure_annotations
def save_pkl(data:dict, file_path:Path):
    with open(file_path, 'wb') as file:
        pickle.dump(data, file)
        print(f"Pickle file saved to {file_path}")


@ensure_annotations
def load_pkl(file_path:Path):
    with open(file_path, 'rb') as file:
        data = pickle.load(file)
        return data




def save_joblib(data, file_path):
    joblib.dump(data, file_path)
    print(f"Joblib file saved to {file_path}")



def load_joblib(file_path):
    data = joblib.load(file_path)
    return data





def save_model_to_artifacts(experiment_id: str) -> str:
    # Create timestamp string: YYYY-MM-DD_HH-MM-SS
    timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")

    # Construct full path
    save_path = Path("artifacts")/ experiment_id / timestamp
    save_path.mkdir(parents=True, exist_ok=True)  # Create all directories

    # Define full model path
    model_path = save_path

    return str(model_path)
