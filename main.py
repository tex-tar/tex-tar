import os
import wandb
import torch
import torch.optim as optim
import argparse
from src.utils.run_utils import (
    set_seed, load_yaml,load_module,
    load_checkpoint
)
from src.utils import pretrained_helper

from src.training.train import Trainer
from src.validation.val import Validator
from src.utils.helper import initialize_optimizer

device = "cuda" if torch.cuda.is_available() else "cpu"

class Pipeline:
    def __init__(self, cfg_path='config/model_config.yaml'):
        set_seed(21)
        self.cfg = load_yaml(cfg_path)
        os.environ['WANDB_DIR'] = self.cfg.get('wandb_dir', os.getenv('WANDB_DIR',''))
        self.checkpoint_dir     = self.cfg['checkpoint_dir']
        os.makedirs(self.checkpoint_dir, exist_ok=True)
        self.purpose = self.cfg['purpose']
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
            pretrained_func = self.cfg.get('pretrained_function',None)
            if pretrained_func!=None and type(pretrained_func)!=bool:
                pretrained_func = getattr(pretrained_helper,pretrained_func)
            epoch, tr_loss, val_loss = load_checkpoint(
                self.model, self.cfg['pretrained'], pretrained_func
            )
            print(f"Resumed from epoch {epoch}, train loss={tr_loss}, val loss={val_loss}")
            return epoch
        return 0

    def run(self):
        start_ep = self.start_epoch
        best_avg_accuracy = 0
        min_loss=1e9
        max_f1=0
        for e in range(start_ep, self.cfg['epochs']):
            train_loss = self.trainer.run_epoch(self.model,self.optimizer,e)
            val_loss,val_accs, f1_report = self.validator.validate(self.model)
            print(f"Epoch : {e} :: Train loss : {train_loss} Val loss : {val_loss}")
            avg_acc=0
            
            for cat_id in val_accs:
                print(f"Val Accuracy for Type {cat_id+1} :",val_accs[cat_id])
                avg_acc+=val_accs[cat_id]
            avg_acc /=len(list(val_accs.keys()))
            if best_avg_accuracy<=avg_acc:
                print(f"UPDATED at epoch {e}")
                best_avg_accuracy = avg_acc
                self.save_model(epoch_id=e,train_loss=train_loss,val_loss=val_loss,type="best_avg_acc",save_name=self.cfg['model'])
            if min_loss>=val_loss:
                min_loss = val_loss
                self.save_model(epoch_id=e,train_loss=train_loss,val_loss=val_loss,type="best_loss",save_name=self.cfg['model'])
            if max_f1<=f1_report:
                max_f1 = f1_report
                self.save_model(epoch_id=e,train_loss=train_loss,val_loss=val_loss,type="best_f1_textar_eval",save_name=self.cfg['model'])
                
            print("F1 score report :", f1_report)
            if self.cfg['use_wandb']:
                wandb.log({"train_loss":train_loss, "val_loss":val_loss, "val_avg_acc":avg_acc})
    
    def eval(self):
        val_loss,val_accs, f1_report = self.validator.validate(self.model)
        avg_acc=0.0
        for cat_id in val_accs:
            avg_acc+=val_accs[cat_id]
        avg_acc /=len(list(val_accs.keys()))
        
        print(f"Test loss : {val_loss} :: Test accuracy : {avg_acc}")
        print(f"Test F1-score : {f1_report}")
    
if __name__=="__main__":
    cfg_path='config/model_config.yaml'
    pipeline = Pipeline(cfg_path)
    parser = argparse.ArgumentParser(description="TexTAR Pipeline")
    parser.add_argument(
        "--config","-c",
        default="config/model_config.yaml",
        help="path to model_config.yaml",
    )

    args = parser.parse_args()
    if args.mode:
      pipeline.purpose = args.mode

  # 3) dispatch
    if pipeline.purpose == "train":
        pipeline.run()
    elif pipeline.purpose == "test":
        pipeline.eval()      # or you can introduce a separate test() if needed
    else:
        parser.error("unknown mode, must be one of train/test")