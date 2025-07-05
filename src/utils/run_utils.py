import importlib
import yaml
import random
import numpy as np
import torch

def set_seed(seed: int):
    """Make training reproducible."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark     = False

def load_yaml(path: str) -> dict:
    with open(path, 'r') as f:
        cfg = yaml.safe_load(f)
    return cfg

def load_module(module_name: str, pkg: str):
    """Dynamically import `utils.<pkg>.<module_name>` and return it."""
    return importlib.import_module(f"src.{pkg}.{module_name}")

def load_checkpoint(model: torch.nn.Module, path: str, pretrained_func):
    ckpt = torch.load(path, map_location='cpu')
    state_dict = ckpt['model_state_dict']
    if pretrained_func!=None:
        pretrained_func(model,state_dict)
    else:
        model.load_state_dict(state_dict)
    return ckpt['epoch'], ckpt['train_loss'], ckpt['val_loss']

def weighted_avg(f1_dict, weights):
    total = sum(weights.values())
    return sum(f1_dict[k] * (w/total) for k, w in weights.items())

def calculate_weighted_average(t1_f1, t2_f1):
    t1_w = {'no_bi':1,'bold':6,'italic':2,'b+i':4}
    t2_w = {'no_us':1,'underlined':6,'strikeout':1,'u+s':1}
    return weighted_avg(t1_f1, t1_w), weighted_avg(t2_f1, t2_w)