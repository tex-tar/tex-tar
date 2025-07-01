# train.py

import os
import yaml
import torch
import importlib
import wandb
from torch.utils.data import DataLoader
from torch.optim.lr_scheduler import ReduceLROnPlateau

# your project–specific imports
from src.dataloader.transform import loader_transform_train
from src.dataloader.samplers import make_sampler
from src.dataloader.dataloader import *
from src.utils.helper import repeat_dataloader

class Trainer:
    def __init__(self, cfg: dict):
        self.cfg = cfg
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


        self.sequence_size = cfg['sequence_size'][0] if isinstance(cfg['sequence_size'], (list,tuple)) else cfg['sequence_size']
        self.model = self._build_model(cfg['model']).to(self.device)

        self.optimizer = torch.optim.Adam(
            self.model.parameters(),
            lr=cfg['lr']
        )


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
        self.loss_types   = cfg['loss_type']['train']
        self.loss_weights = cfg['loss_weights']

        # ---- 5) data loaders ----
        self.train_loader, self.val_loader = self._init_dataloaders(cfg)

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

    def _build_model(self, model_name: str) -> torch.nn.Module:
        """
        Given a string like "consent_rope_selective", import
        utils.architectures.consent_rope_selective and call its
        `model(...)` factory.
        """
        mod = importlib.import_module(f"utils.architectures.{model_name}")
        if hasattr(mod, 'model'):
            try:
                return mod.model(sequence_size=self.sequence_size)
            except TypeError:
                return mod.model()
        raise RuntimeError(f"architecture module {model_name} has no `model(...)`")

    def _init_dataloaders(self, cfg):
        """
        Builds a zipped list of DataLoaders, one per dataset in cfg['datasets'].
        """
        loaders, lengths = [], []
        for ds_cfg, sam_cfg in zip(cfg['datasets'], cfg['samplers']):
            # dynamically load the Dataset class
            DS = getattr(importlib.import_module("src.dataloader.dataloader"), ds_cfg['class'])
            ds = DS(
                img_dir=ds_cfg['img_dir'],
                order_list=ds_cfg['order_list'],
                label_split=ds_cfg['label_split'],
                total_categories=ds_cfg['total_categories'],
                transform=loader_transform_train
            )

            if sam_cfg.get('use'):
                sampler = make_sampler(
                    sam_cfg['order_list'],
                    sam_cfg['fraction'],
                    sam_cfg['class_weights'],
                    ds_cfg['total_categories']
                )
            else:
                sampler = None

            loader = DataLoader(
                ds,
                batch_size=ds_cfg['batch_size'],
                shuffle=(sampler is None and ds_cfg['shuffle']),
                sampler=sampler,
                num_workers=cfg['num_workers'],
                pin_memory=True
            )
            loaders.append(loader)
            lengths.append(len(loader))

        # equalize lengths by repeating shorter loaders
        max_len = max(lengths)
        final = []
        for L, length in zip(loaders, lengths):
            if length < max_len:
                final.append(repeat_dataloader(L, max_len))
            else:
                final.append(L)

        # zip them for synchronized iteration
        return list(zip(*final))

    def train(self, epochs: int):
        for epoch in range(1, epochs+1):
            train_loss = self._run_epoch(epoch)
            if self.scheduler:
                self.scheduler.step(train_loss)
            # you could call a validation pass here, save checkpoints, etc.

    def _run_epoch(self, epoch: int) -> float:
        self.model.train()
        self.optimizer.zero_grad()
        total_loss = 0.0
        batch_count = 0

        for batch in self.train_loader:
            # each batch is a tuple of sub-batches, one per loader
            # here we assume 3-tuple: (images, labels, supp)
            # adjust this unpacking to your datasets
            images, labels, supp = batch  
            images = images.to(self.device)
            labels = labels.to(self.device)
            supp   = supp.to(self.device)

            outputs = self.model(images, supp)
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

        avg = total_loss / batch_count
        print(f"[Epoch {epoch}] train loss = {avg:.4f}")
        return avg


def main():
    # assume your config lives in `config.yaml`
    with open("config/config.yaml") as f:
        cfg = yaml.safe_load(f)

    trainer = Trainer(cfg)
    trainer.train(cfg['epochs'])


if __name__ == "__main__":
    main()