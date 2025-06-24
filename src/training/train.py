import os
import time
import yaml
import torch
import wandb

from torch.utils.data import DataLoader
from torch.optim.lr_scheduler import ReduceLROnPlateau

from utils.dataloader.transform import loader_transform_train
from utils.dataloader.samplers import make_sampler
from utils.dataloader.dataloader import *
from utils.helper import load_module, repeat_dataloader
from utils.files.pretrained_helper import load_pretrained

class Trainer:
    def __init__(self, cfg: dict):
        # Model and optimization
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.model = self._build_model(cfg['model']).to(self.device)
        self.optimizer = self._build_optimizer(cfg['optimizer'], self.model)
        self.scheduler = self._build_scheduler(cfg.get('scheduler'), self.optimizer)

        # Loss
        loss_mod = load_module(cfg['loss_function_module'], 'loss_functions')
        self.calculate_loss = loss_mod.calculate_loss
        self.loss_types = cfg['loss_types']
        self.loss_weights = cfg['loss_weights']

        # Data loaders
        self.init_dataloaders = self._init_dataloaders(cfg['data'])
        self.accumulation_steps = cfg['training']['accumulation_steps']

        # Checkpoint and logging
        self.checkpoint_dir = cfg['training']['checkpoint_dir']
        os.makedirs(self.checkpoint_dir, exist_ok=True)

        if cfg['training'].get('use_wandb'):
            wandb.init(
                project=cfg['training']['wandb_project'],
                name=cfg['training']['run_name'],
                config=cfg
            )

    def _build_model(self, model_cfg: dict) -> torch.nn.Module:
        # instantiate your model here, e.g. vision+transformer
        ModelClass = load_module(model_cfg['module'], model_cfg['class'])
        model = ModelClass(**model_cfg.get('params', {}))
        if model_cfg.get('pretrained_path'):
            load_pretrained(model, model_cfg['pretrained_path'], model_cfg.get('load_fn'))
        return model

    def _build_optimizer(self, optim_cfg: dict, model: torch.nn.Module):
        OptimClass = getattr(torch.optim, optim_cfg['type'])
        return OptimClass(model.parameters(), **optim_cfg['params'])

    def _build_scheduler(self, sched_cfg: dict, optimizer):
        if not sched_cfg:
            return None
        if sched_cfg['type'] == 'ReduceLROnPlateau':
            return ReduceLROnPlateau(optimizer, **sched_cfg.get('params', {}))
        # add more schedulers as needed
        return None

    def _init_dataloaders(self, data_cfg: dict):
        dataloaders = []
        lengths = []
        for ds_cfg, sampler_cfg in zip(data_cfg['datasets'], data_cfg['samplers']):
            DatasetClass = load_module(ds_cfg['module'], ds_cfg['class'])
            dataset = DatasetClass(
                img_dir=ds_cfg['img_dir'],
                order_list=ds_cfg['order_list'],
                label_split=ds_cfg['label_split'],
                total_categories=ds_cfg['total_categories'],
                transform=loader_transform_train
            )
            if sampler_cfg['use']:
                sampler = make_sampler(
                    sampler_cfg['order_list'],
                    sampler_cfg['fraction'],
                    sampler_cfg['class_weights'],
                    ds_cfg['total_categories']
                )
            else:
                sampler = None

            loader = DataLoader(
                dataset,
                batch_size=ds_cfg['batch_size'],
                shuffle=(sampler is None and ds_cfg['shuffle']),
                sampler=sampler,
                num_workers=data_cfg['num_workers'],
                pin_memory=True,
                worker_init_fn=seed_worker
            )
            dataloaders.append(loader)
            lengths.append(len(loader))

        # repeat shorter loaders to match longest
        max_len = max(lengths)
        final_loaders = []
        for loader, length in zip(dataloaders, lengths):
            if length < max_len:
                final_loaders.append(repeat_dataloader(loader, max_len))
            else:
                final_loaders.append(loader)
        return list(zip(*final_loaders))  # for synchronized iteration

   

    def _run_train(self, epoch):
        self.model.train()
        self.optimizer.zero_grad()
        total_loss, count = 0.0, 0
        for batch in self.init_dataloaders:
            inputs, labels, supp = self._unpack_batch(batch)
            outputs = self.model(inputs, supp)
            loss = self.calculate_loss(
                outputs, labels, self.loss_types, self.loss_weights,
                sum(ds['total_categories'] for ds in self.init_dataloaders[0]),
                self.init_dataloaders[0][0].dataset.sequence_size,
                ds['label_split'], [len(inputs)]
            ) / self.accumulation_steps
            loss.backward()
            total_loss += loss.item()
            count += 1
            if count % self.accumulation_steps == 0:
                self.optimizer.step()
                self.optimizer.zero_grad()
        avg_loss = total_loss / count
        print(f"Epoch {epoch} train loss: {avg_loss:.4f}")
        return avg_loss


def main():
    path = "config.yaml"
    with open(path, 'r') as f:
        cfg_data = yaml.safe_load(f)
    cfg = cfg_data['trainer']
    trainer = Trainer(cfg)
    trainer.train(cfg['training']['epochs'])


if __name__ == '__main__':
    main()
