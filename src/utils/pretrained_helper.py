import torch
import torch.nn as nn

def freeze_backbone(model : nn.Module,state_dict):
    model.load_state_dict(state_dict,strict=False)
    for k,v in model.named_parameters():
        if k[:6] in ['model1','model2']:
            print("Freezing layers",k)
            v.requires_grad=False
    return model