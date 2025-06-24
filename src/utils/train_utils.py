import torch
def save_model(self, epoch_id, train_loss, val_loss,type:str,save_name):
        # print(self.model)
        torch.save({
            'epoch': epoch_id+1,
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'train_loss': train_loss,
            'val_loss': val_loss,
        }, f'/ssd_scratch/rohan.kumar/swaroopajinka/checkpoints/{save_name}-{type}.pt')