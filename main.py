import os
import wandb
import torch
import torch.optim as optim
from src.utils.run_utils import (
    set_seed, load_yaml,load_module,
    load_checkpoint
)

from src.training.train import Trainer
from src.validation.val import Validator

device = "cuda" if torch.cuda.is_available() else "cpu"

def initialize_optimizer(optimizer_name, learning_rate, parameters,model=None):
    optimizer_name = optimizer_name.lower()
    if optimizer_name == 'adam':
        optimizer = optim.Adam(parameters, lr=learning_rate)
    elif optimizer_name == 'adam_mtl':
        lrlast = .001
        lrmain = .0001
        optimizer = optim.Adam([
            {"params": model.resnet_model.parameters(), "lr": lrmain},
            {"params": model.x1.parameters(), "lr": lrlast},
            {"params": model.x2.parameters(), "lr": lrlast},
            {"params": model.y1o.parameters(), "lr": lrlast},
            {"params": model.y2o.parameters(), "lr": lrlast},
        ])

    elif optimizer_name == 'sgd':
        optimizer = optim.SGD(parameters, lr=learning_rate)
    elif optimizer_name == 'adadelta':
        optimizer = optim.Adadelta(parameters, lr=learning_rate)
    elif optimizer_name == 'rmsprop':
        optimizer = optim.RMSprop(parameters, lr=learning_rate)        
    else:
        raise ValueError(f"Optimizer '{optimizer_name}' is not supported.")
    
    return optimizer

class Pipeline:
    def __init__(self, cfg_path='config/train_config.yaml'):
        set_seed(21)
        self.cfg = load_yaml(cfg_path)
        os.environ['WANDB_DIR'] = self.cfg.get('wandb_dir', os.getenv('WANDB_DIR',''))
        self.checkpoint_dir     = self.cfg['checkpoint_dir']
        os.makedirs(self.checkpoint_dir, exist_ok=True)
        
        self._build_architecture()
        
        self.start_epoch = self._load_pretrained()
        if self.cfg['use_wandb']:
            wandb.init(
                project=self.cfg['wandb_project_name'],
                name   =self.cfg['run_name'],
                reinit =True
            )
            
    def save_model(self,epoch_id, train_loss, val_loss,type:str,save_name):
            torch.save({
                'epoch': epoch_id+1,
                'model_state_dict': self.model.state_dict(),
                'optimizer_state_dict': self.optimizer.state_dict(),
                'train_loss': train_loss,
                'val_loss': val_loss,
            }, os.path.join(self.checkpoint_dir,f'{save_name}-{type}.pt'))
        
    def _build_architecture(self):
        mmod = load_module(self.cfg['model'], 'architectures')
        try:
            self.model = mmod.model(sequence_size=self.cfg['sequence_size'])
        except TypeError:
            self.model = mmod.model
            
        self.model = self.model.to(device)
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
        best_avg_accuracy = 0
        avg_acc=0
        for e in range(start_ep, self.cfg['epochs']):
            train_loss = self.trainer.run_epoch(self.model,self.optimizer,e)
            val_loss,val_accs = self.validator.validate(self.model)
            print(f"Epoch : {e} :: Train loss : {train_loss} Val loss : {val_loss}")
            for cat_id in val_accs:
                print(f"Val Accuracy for Type {cat_id+1} :",val_accs[cat_id])
                avg_acc+=val_accs[cat_id]
            avg_acc /=len(val_accs)
            if best_avg_accuracy<avg_acc:
                best_avg_accuracy = avg_acc
                self.save_model(epoch_id=e,train_loss=train_loss,val_loss=val_loss,type="best_avg_acc",save_name="textar-base")
            

if __name__=="__main__":
    Pipeline().run()