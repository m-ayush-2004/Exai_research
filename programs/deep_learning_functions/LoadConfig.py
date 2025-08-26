import os
import json
from dotenv import load_dotenv
load_dotenv()

DL_CONFIG_FILE_PATH = os.getenv("DL_CONFIG_FILE_PATH")

# Load available models from config file
def load_config(config_path=DL_CONFIG_FILE_PATH):
    if os.path.exists(config_path):
        with open(config_path, 'r') as f:
            config = json.load(f)
        return config['models']
    return {}

# Save model configuration to the config file
def save_model(models, config_path=DL_CONFIG_FILE_PATH):
    with open(config_path, 'w') as f:
        json.dump({"models": models}, f, indent=4)