import os
import yaml
import json
import time
import torch
import numpy as np
from torch.utils.data import DataLoader
from src.dataloader import dataloader
from src.dataloader.transform import loader_transform_val
from src.utils.helper import initialize_loss
from src.inference.calc_report import evaluate_textar
import importlib

class Validator:
    def __init__(self, cfg: dict):
        self.cfg = cfg
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.sequence_size = cfg['sequence_size'][0] if isinstance(cfg['sequence_size'], (list,tuple)) else cfg['sequence_size']
        self.label_split = cfg['label_split']
        
        # ---- 4) loss fn ----
        loss_mod = importlib.import_module(f"src.loss_functions.{cfg['loss_fn']['val']}")
        self.calculate_loss = loss_mod.calculate_loss
        self.loss_types   = [initialize_loss(cfg['loss_types']['val'][idx]) for idx in range(len(cfg['loss_types']['val']))]
        self.loss_weights = cfg['loss_weights']

        # ---- 5) data loaders ----
        self.val_dataset_class = getattr(dataloader,cfg['dataloader']['val'])
        self.val_dataset = self.val_dataset_class(
            labels_bbox_json_path=cfg['bounding_box_label_jsons']['val'],
            img_dir=cfg['datasets']['val'],
            label_split=cfg['label_split'],
            total_categories=len(cfg['label_split']),
            transform=loader_transform_val
        )
        
        # set val batch size as 8
        self.val_loader = DataLoader(self.val_dataset,batch_size=8,shuffle=False,collate_fn=self.collate_func,num_workers=cfg["num_workers"],pin_memory=True)
    
    def collate_func(self,batch):
        imgs, labels, infos = zip(*batch)  # unzip the batch
        imgs = torch.stack(imgs)
        labels = torch.stack(labels)
        imgs = imgs.view(-1,imgs.size(-3),imgs.size(-2),imgs.size(-1))        
        labels = labels.view(-1,labels.size(-1))
        
        merged_info = {}
        for key in infos[0].keys():
            merged_info[key] = []
            for info in infos:
                merged_info[key].extend(info[key])  # concatenate lists

        return imgs, labels, merged_info

    def validate(self, model):
        model.eval()
        total_loss = 0.0
        batch_count = 0
        preds_cats = {}        
        preds_cats_logits = {}
        labels_cats = {}
        imgs_names_cats = []

        for batch in self.val_loader:
            images, labels, supp = batch  
            images = images.to(self.device)
            labels = labels.to(self.device)
            labels = labels.view(-1,labels.size(-1))
            
            outputs = model(images, supp)
            imgs_names_cats.extend(supp['img_files'])

            # divide the packed tensor based on the categories in config file
            cum_label_ctr=0
            for cat_id, cat_lbl_var in enumerate(self.cfg["label_split"]):
                val = preds_cats.get(cat_id,-1)
                if val==-1:
                    preds_cats[cat_id]=torch.argmax(outputs[:,cum_label_ctr:cat_lbl_var+cum_label_ctr],dim=-1).cpu().tolist()
                    preds_cats_logits[cat_id]=outputs[:,cum_label_ctr:cat_lbl_var+cum_label_ctr].cpu().tolist()
                    labels_cats[cat_id]=torch.argmax(labels[:,cum_label_ctr:cat_lbl_var+cum_label_ctr],dim=-1).cpu().tolist()
                else:
                    preds_cats[cat_id].extend(torch.argmax(outputs[:,cum_label_ctr:cat_lbl_var+cum_label_ctr],dim=-1).cpu().tolist())
                    preds_cats_logits[cat_id].extend(outputs[:,cum_label_ctr:cat_lbl_var+cum_label_ctr].cpu().tolist())
                    labels_cats[cat_id].extend(torch.argmax(labels[:,cum_label_ctr:cat_lbl_var+cum_label_ctr],dim=-1).cpu().tolist())
                cum_label_ctr+=cat_lbl_var
                
            loss = self.calculate_loss(
                outputs,
                labels,
                self.loss_types,
                self.loss_weights,
                self.label_split
            ) 

            total_loss += loss.item()
            batch_count += 1
        # print(torch.bincount(torch.tensor(labels_cats[1])))        
        accuracies = {}
        # accuracy for each category T1 and T2
        for k,v in preds_cats.items():
            sum = torch.sum(torch.tensor(preds_cats[k])==torch.tensor(labels_cats[k]))
            total = len(preds_cats[k])
            accuracy = (sum/total)*100
            accuracies[k] = accuracy

        f1_agg_report,macro_F1_overall = evaluate_textar(preds_all=preds_cats_logits,imgs_all=imgs_names_cats,bbox_info_labels=self.val_dataset.bbjson,root_path=self.cfg['datasets']['val'])
        print(f1_agg_report)
        avg_loss = total_loss / batch_count
        return avg_loss, accuracies, macro_F1_overall


