
import json

def load_config(config_path: str) -> dict:
    with open(config_path, 'r') as f:
        return json.load(f)

def load_default_attributes(config_path: str = None) -> dict:
    if config_path:
        with open(config_path, 'r') as f:
            return json.load(f)
    return {}