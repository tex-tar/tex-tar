import os
import yaml
import json
import time
import torch
import numpy as np
from torch.utils.data import DataLoader
from utils.dataloader.transform import loader_transform_val
from utils.helper import load_module, load_config
from utils.run_utils import calculate_weighted_average

class Validator:
    def __init__(self, cfg: dict):
        self.cfg = cfg
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.sequence_size = cfg['sequence_size'][0] if isinstance(cfg['sequence_size'], (list,tuple)) else cfg['sequence_size']
        
        # ---- 4) loss fn ----
        loss_mod = importlib.import_module(f"utils.loss_functions.{cfg['loss_fn']['val']}")
        self.calculate_loss = loss_mod.calculate_loss
        self.loss_types   = cfg['loss_types']['val']
        self.loss_weights = cfg['loss_weights']

        # ---- 5) data loaders ----
        self.val_dataset = getattr(dataloader,cfg['dataloader']['val'])
        
        # set val batch size as 640
        self.val_loader = DataLoader(self.val_dataset,batch_size=640,shuffle=False, num_workers=cfg["num_workers"],pin_memory=True)

    def validate(self, model):
        model.eval()
        total_loss = 0.0
        batch_count = 0
        preds_cats = {}
        labels_cats = {}

        for batch in self.val_loader:
            images, labels, supp = batch  
            images = images.to(self.device)
            labels = labels.to(self.device)
            supp   = supp.to(self.device)

            outputs = model(images, supp)
            
            # divide the packed tensor based on the categories in config file
            cum_label_ctr=0
            for cat_id, cat_lbl_var in enumerate(self.cfg["output_categories"]):
                val = preds_cats.get(cat_id,-1)
                if val==-1:
                    preds_cats[cat_id]=torch.argmax(outputs[:,cum_label_ctr:cat_lbl_var],dim=-1).cpu().tolist()
                    labels_cats[cat_id]=torch.argmax(labels[:,cum_label_ctr:cat_lbl_var],dim=-1).cpu().tolist()
                else:
                    preds_cats[cat_id].extend(torch.argmax(outputs[:,cum_label_ctr:cat_lbl_var],dim=-1).cpu().tolist())
                    labels_cats[cat_id].extend(torch.argmax(labels[:,cum_label_ctr:cat_lbl_var],dim=-1).cpu().tolist())
                cum_label_ctr+=cat_lbl_var

            loss = self.calculate_loss(
                outputs,
                labels,
                self.loss_types,
                self.loss_weights
            )

            total_loss += loss.item()
            batch_count += 1
            
        avg_loss = total_loss / batch_count
        return avg_loss


