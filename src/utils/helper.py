import torch.optim as optim
import torch.nn as nn
import torch.nn.functional as F
import torch

class FocalLoss(nn.Module):
    def __init__(self, alpha=None, gamma=2):
        super(FocalLoss, self).__init__()
        if alpha!=None:
            self.alpha = torch.tensor(alpha).cuda()
        else:
            self.alpha=None
        self.gamma = gamma

    def forward(self, inputs, targets):
        # BCE_loss = F.binary_cross_entropy_with_logits(inputs, targets, reduction='none')
        # targets = targets.type(torch.long)
        labels = torch.argmax(targets,1)
        at = self.alpha.gather(0,labels)
        CE_loss = torch.nn.functional.cross_entropy(inputs, targets, reduction='none')
        pt = torch.exp(-CE_loss)
        return (at*(1-pt)**self.gamma * CE_loss).mean() / self.alpha.sum()

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

def initialize_loss(loss_name):
    loss_name = loss_name.lower()
    if loss_name == 'ce':
        loss_function = nn.CrossEntropyLoss()
    elif loss_name == 'bce':
        loss_function = nn.BCELoss()
    elif loss_name == 'mse':
        loss_function = nn.MSELoss()
    elif loss_name == 'train_wce':
        class_weights =torch.tensor([1,4,2,6],dtype=torch.float32)
        class_weights=class_weights.cuda()
        loss_function = nn.CrossEntropyLoss(weight=class_weights)
    elif loss_name == 'train_wce-t1':
        class_weights =torch.tensor([2,3,2,6],dtype=torch.float32)
        class_weights=class_weights.cuda()
        loss_function = nn.CrossEntropyLoss(weight=class_weights)
    elif loss_name == 'train_focal':
        #non weighted
        loss_function = FocalLoss(alpha=[1.0555,28.6700,114.5004,111.6065],gamma=2)
    elif loss_name == 'val_focal':
        #non weighted
        loss_function = FocalLoss(alpha=[1.0430,37.6126,131.7034,140.3361],gamma=2)
    else:
        raise ValueError(f"Loss function '{loss_name}' is not supported.")
    
    return loss_function