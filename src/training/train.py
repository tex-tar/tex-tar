import os
import yaml
import torch
import importlib
import wandb
from torch.utils.data import DataLoader
from torch.optim.lr_scheduler import ReduceLROnPlateau
from src.utils.dataloader import dataloader
from src.dataloader.transform import loader_transform_train
from src.dataloader.dataloader import *
from src.utils.helper import repeat_dataloader

class Trainer:
    def __init__(self, cfg: dict):
        self.cfg = cfg
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.sequence_size = cfg['sequence_size'][0] if isinstance(cfg['sequence_size'], (list,tuple)) else cfg['sequence_size']
        
        ## you can also specify a scheduler if you want
        sched_cfg = cfg.get('scheduler')
        if sched_cfg and sched_cfg.get('type') == 'ReduceLROnPlateau':
            self.scheduler = ReduceLROnPlateau(
                self.optimizer,
                **sched_cfg.get('params', {})
            )
        else:
            self.scheduler = None

        # ---- 4) loss fn ----
        loss_mod = importlib.import_module(f"utils.loss_functions.{cfg['loss_fn']['train']}")
        self.calculate_loss = loss_mod.calculate_loss
        self.loss_types   = cfg['loss_types']['train']
        self.loss_weights = cfg['loss_weights']

        # ---- 5) data loaders ----
        self.train_dataset = getattr(dataloader,cfg['dataloader']['train'])
        self.train_loader = DataLoader(self.train_dataset,batch_size=cfg["batch_size"],shuffle=cfg["shuffle"], num_workers=cfg["num_workers"],pin_memory=True)
    
        # ---- 6) misc ----
        self.accumulation_steps = cfg['accumulated_batch_descent']
        self.checkpoint_dir     = cfg['checkpoint_dir']
        os.makedirs(self.checkpoint_dir, exist_ok=True)

        if cfg.get('use_wandb'):
            wandb.init(
                project=cfg['wandb_project_name'],
                name=cfg['run_name'],
                config=cfg
            )

    def run_epoch(self, model, optimizer, epoch: int) -> float:
        model.train()
        optimizer.zero_grad()
        total_loss = 0.0
        batch_count = 0

        for batch in self.train_loader:
            images, labels, supp = batch  
            images = images.to(self.device)
            labels = labels.to(self.device)
            supp   = supp.to(self.device)

            outputs = model(images, supp)
            loss = self.calculate_loss(
                outputs,
                labels,
                self.loss_types,
                self.loss_weights
            ) / self.accumulation_steps

            loss.backward()
            total_loss += loss.item()
            batch_count += 1

            if batch_count % self.accumulation_steps == 0:
                self.optimizer.step()
                self.optimizer.zero_grad()

        avg_loss = total_loss / batch_count
        return avg_loss