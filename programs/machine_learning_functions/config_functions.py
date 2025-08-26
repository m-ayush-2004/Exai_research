import os
import json
from dotenv import load_dotenv

ML_CONFIG_FILE_PATH = os.getenv('ML_CONFIG_FILE_PATH')
def load_model_config():
    """
    The `load_model_config` function reads a JSON file containing model configurations specified by the
    environment variable `ML_CONFIG_FILE_PATH`.
    
    Returns:
      The function `load_model_config()` returns the loaded models configuration from the specified file
    path.
    """
    ML_CONFIG_FILE_PATH = os.getenv('ML_CONFIG_FILE_PATH')
    with open(ML_CONFIG_FILE_PATH, 'r') as file:
        models_config = json.load(file)
    print(f"\033[32m[INFO::]\033[00mLoaded models configs from {ML_CONFIG_FILE_PATH}")  # Add this line for debugging
    return models_config

def update_config_file(model_name, target_column, feature_columns):
    config_data = {}
    if os.path.exists(ML_CONFIG_FILE_PATH):
        with open(ML_CONFIG_FILE_PATH, 'r') as file:
            config_data = json.load(file)
    config_data[model_name] = {'target_column': target_column, 'features': feature_columns}
    with open(ML_CONFIG_FILE_PATH, 'w') as file:
        json.dump(config_data, file, indent=4)
