import os
import wandb
import torch

from utils.run_utils import (
    set_seed, load_json, load_module,
    load_checkpoint, calculate_weighted_average
)

from src.utils.helper import initialize_optimizer
from src.training.train import Trainer
from src.validation.val import Validator

class Pipeline:
    def __init__(self, cfg_path='config/train_config.yaml'):
        set_seed(21)
        self.cfg = load_json(cfg_path)
        os.environ['WANDB_DIR'] = self.cfg.get('wandb_dir', os.getenv('WANDB_DIR',''))
        self._build_architecture()
        self.start_epoch = self._load_pretrained()
        if self.cfg['isWandb']:
            wandb.init(
                project=self.cfg['wandb_project_name'],
                name   =self.cfg['run_name'],
                reinit =True
            )
    
    def _build_architecture(self):
        mmod = load_module(self.cfg['model'], 'architectures')
        try:
            self.model = mmod.model(sequence_size=self.cfg['sequence_size'])
        except TypeError:
            self.model = mmod.model

        opt_cfg = self.cfg['optimizer']

        self.optimizer = initialize_optimizer(self.cfg["optimizer"],learning_rate=self.cfg['lr'],parameters = self.model.parameters())
        
        self.trainer = Trainer(self.cfg)
        self.validator = Validator(self.cfg)

    def _load_pretrained(self):
        if self.cfg['pretrained']:
            epoch, tr_loss, val_loss = load_checkpoint(
                self.model, self.cfg['pretrained']
            )
            print(f"Resumed from epoch {epoch}, train loss={tr_loss}, val loss={val_loss}")
            return epoch
        return 0

    def run(self):
        start_ep = self.start_epoch
        for e in range(start_ep, self.cfg['epochs']):
            train_loss = self.trainer.run_epoch(self.model,self.optimizer,e)
            val_acc,val_f1,val_loss = self.validator.validate(self.model)
            print(f"Epoch : {e} :: Train loss : {train_loss} Val loss : {val_loss} Val accuracy : {val_acc} Val F1-score : {val_f1}")

if __name__=="__main__":
    Pipeline().run()