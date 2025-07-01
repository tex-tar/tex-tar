from torch.utils.data.sampler import WeightedRandomSampler
import numpy as np 
import random 
import pandas as pd   
import torch

def make_weights(train_targets, prob_sample,num_classes):
    class_sample_counts = [train_targets.count(class_idx) for class_idx in range(num_classes)]
    weights = np.array(class_sample_counts, dtype=np.float32)

    class_weights = np.array(prob_sample) / weights
    # import pdb; pdb.set_trace()
    train_targets = np.array(train_targets, dtype=np.float64)
    for idx in range(num_classes):
        train_targets[np.where(np.array(train_targets)==idx)[0]] = class_weights[idx]
    # Define sampler for weighted sampling on the training set
    train_targets = torch.tensor(train_targets) 

    return train_targets

# load dataframe
def make_sampler(csv_path_sampler,fraction,weights,num_classes):
    df = pd.read_csv(csv_path_sampler)
    train_targets = df['label'].tolist()
    train_targets = make_weights(train_targets, prob_sample=weights,num_classes=num_classes)
    train_targets = train_targets.to("cuda")
    sampler = WeightedRandomSampler(train_targets, int(fraction*train_targets.shape[0]), replacement=True)
    return sampler