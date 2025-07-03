import torch

def calculate_loss(outputs, input_labels, loss_criterions, loss_weights, label_split, seq_size, mask):
    splitter_idx = 0
    batch_splitter = 0
    
    final_loss = torch.tensor(0.0)
    for idx in range(len(label_split)):
        if mask[idx]==-1:
            splitter_idx += label_split[idx]
            continue
        elif mask[idx]==0:
            continue
        else:
            outputs_subset = outputs[:,splitter_idx:splitter_idx+label_split[idx]]
            input_subset = input_labels[:,splitter_idx:splitter_idx+label_split[idx]]
            loss_calculated = loss_criterions[idx](outputs_subset,input_subset)
            final_loss = final_loss + loss_weights[idx]*loss_calculated
            splitter_idx += label_split[idx]
            
    return final_loss