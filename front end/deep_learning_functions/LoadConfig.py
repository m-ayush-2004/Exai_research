import os
import json
# Load available models from config file
def load_model_config(config_path='front end/config/config2.json'):
    if os.path.exists(config_path):
        with open(config_path, 'r') as f:
            config = json.load(f)
        return config.get('models', {})
    return {}
