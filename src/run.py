import os
import wandb
import torch

from utils.run_utils import (
    set_seed, load_json, load_module,
    load_checkpoint, calculate_weighted_average
)
from training.train import Trainer

class Pipeline:
    def __init__(self, cfg_path='config.json'):
        set_seed(21)
        self.cfg = load_json(cfg_path)
        os.environ['WANDB_DIR'] = self.cfg.get('wandb_dir', os.getenv('WANDB_DIR',''))
        self._build_architecture()
        self._load_pretrained()
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
        Optim   = getattr(torch.optim, opt_cfg['type'])
        self.optimizer = Optim(self.model.parameters(),lr=self.cfg['lr'],**opt_cfg.get('params',{}))

        Trainer   = load_module('train_added','files').Trainer
        Validator = load_module('val_added','files').Validator
        Visualize = load_module('test','files').visualize_results
        common = {
            'model':   self.model,
            'optimizer': self.optimizer,
            'loss_types': self.cfg['loss_type'],
            'loss_weights': self.cfg['loss_weights'],
            'label_split': self.cfg['label_split'],
            'batch_size': self.cfg['batch_size'],
            'batch_split': self.cfg['batch_split'],
            'accumulated_batch_descent': self.cfg['accumulated_batch_descent'],
            'sequence_size': self.cfg['sequence_size'],
            'num_workers': self.cfg['num_workers'],
            'total_categories': self.cfg['output_categories'],
        }

        
        self.trainer = Trainer(
            **common,
            datasets=self.cfg['datasets']['train'],
            isWandb=self.cfg['isWandb'],
            order_files=self.cfg['order']['train'],
            dloader_types=self.cfg['dataloaders']['train'],
            loss_function=self.cfg['loss_fn']['train'],
            load_pretrained_function=self.cfg['load_pretrained_function'],
            use_external_sampler=self.cfg['use_external_sampler'],
            sampler_fraction_dataset=self.cfg['sampler_dataset_fraction'],
            sampler_class_weights=self.cfg['sampler_class_weights'],
            shuffle_array=self.cfg['shuffle']
        )
        self.validator = Validator(
            **common,
            datasets=self.cfg['datasets']['val'],
            isWandb=self.cfg['isWandb'],
            order_files=self.cfg['order']['val'],
            dloader_types=self.cfg['dataloaders']['val'],
            loss_function=self.cfg['loss_fn']['val'],
            shuffle_array=self.cfg['shuffle']
        )
        self.visualize = Visualize

    def _load_pretrained(self):
        if self.cfg['pretrained']:
            epoch, tr_loss, val_loss = load_checkpoint(
                self.model, self.cfg['pretrained']
            )
            print(f"Resumed from epoch {epoch}, train loss={tr_loss}, val loss={val_loss}")
            return epoch
        return 0

    def run(self):
        purpose = self.cfg['purpose']
        start_ep = 0
        if purpose != 'test':
            best = {'t1':0., 't2':0., 'avg':0., 'val':1e9}
       
            for e in range(start_ep, self.cfg['epochs']):
                _, train_loss = self.trainer.run_epoch(e)
                val_loss, t1_f1, t2_f1 = self.validator.eval(e, self.model, bbox_info=None)
                w1, w2 = calculate_weighted_average(t1_f1, t2_f1)
                avg_f1 = (w1 + w2)/2

                
                if w1 > best['t1']:
                    best['t1'], _ = w1, self.trainer.save_model(e, train_loss, val_loss, 'best_t1_f1')
                if w2 > best['t2']:
                    best['t2'], _ = w2, self.trainer.save_model(e, train_loss, val_loss, 'best_t2_f1')
                if avg_f1 > best['avg']:
                    best['avg'], _ = avg_f1, self.trainer.save_model(e, train_loss, val_loss, 'best_avg_f1')
                if val_loss < best['val']:
                    best['val'], _ = val_loss, self.trainer.save_model(e, train_loss, val_loss, 'best_val')
                self.trainer.save_model(e, train_loss, val_loss, 'last')

                if self.cfg['isWandb']:
                    wandb.log({'train_loss': train_loss,'val_loss': val_loss,'t1_f1': w1,'t2_f1': w2,'avg_f1': avg_f1})
                                   
        else:
            
            Visualize = self.visualize
            Visualize(self.model, self.cfg)

if __name__=="__main__":
    Pipeline().run()