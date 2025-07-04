import torch


def calculate_loss(outputs, input_labels, loss_criterions, loss_weights, label_split):
    input_labels = input_labels.view(-1,input_labels.size(-1))
    final_loss = torch.tensor(0.0)
    idx=0
    temperature=0.25
    splitter_idx = 0
    while idx<len(label_split):
        outputs_subset = outputs[:, splitter_idx:splitter_idx+label_split[idx]]
        outputs_subset = outputs_subset/temperature
        input_subset = input_labels[:, splitter_idx:splitter_idx+label_split[idx]]
        loss_calculated = loss_criterions[idx](outputs_subset,input_subset)
        final_loss = final_loss + loss_weights[idx]*loss_calculated
        splitter_idx += label_split[idx]
        idx+=1
    
    return final_loss
 
 
