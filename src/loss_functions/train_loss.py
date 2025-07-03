import torch


def calculate_loss(outputs, input_labels, loss_criterions, loss_weights, label_split, seq_size, batch_label_split,batch_split):
    input_labels = input_labels.view(-1,input_labels.size(-1))
    final_loss = torch.tensor(0.0)
    final_loss_sq = torch.tensor(0.0)
    idx=0
    # temperature=0.25
    temperature=0.25
    splitter_idx = 0
    for i in batch_label_split[0]:
        # print(outputs.size())
        # print(label_split[idx])
        # print(outputs.shape)
        outputs_subset = outputs[:, splitter_idx:splitter_idx+label_split[idx]]
        #shape of outputs_subset and input_subset is [600,4]
        outputs_subset = outputs_subset/temperature
        input_subset = input_labels[:, splitter_idx:splitter_idx+label_split[idx]]
        splitter_idx += label_split[idx] 
        loss_calculated = loss_criterions[idx](outputs_subset,input_subset)
        final_loss = final_loss + loss_weights[idx]*loss_calculated
        idx+=1
    
    return final_loss
 
 
