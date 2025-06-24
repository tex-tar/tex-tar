from torchvision.models import resnet18
import torch.nn as nn
import functools
import math
import random
import numpy as np
import torch
seed=42
np.random.seed(seed)
random.seed(seed)
# Set the seed for PyTorch
torch.manual_seed(42)
# If using CUDA
if torch.cuda.is_available():
    torch.cuda.manual_seed(42)
    torch.cuda.manual_seed_all(42)  # If using multi-GPU.
    
# For deterministic behavior
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False


embed_dim = 256
num_layers = 6
n_heads = 8

fc_size=256



feature_extractor = resnet18(pretrained=True)
in_features = feature_extractor.fc.in_features
print(in_features)
feature_extractor.fc = nn.Sequential(
    nn.Linear(in_features, 256),
    nn.ReLU()
    # nn.Linear(256, 64)
)

feed_forward = nn.Sequential(
    nn.Linear(256,128), # 64 dim embedding obtained from encoder module
    nn.ReLU(),
    nn.Linear(128,64),
    nn.ReLU(),
    nn.Linear(64,32),
    nn.ReLU(),
    nn.Linear(32,4)
)

class Encoder(nn.Module):
    def __init__(self, emsize=64,nhead=2,n_layers = 4, feed_forward_size=64):
        super(Encoder, self).__init__()
        self.transformer_encoder = nn.TransformerEncoder(nn.TransformerEncoderLayer(d_model=emsize, nhead=nhead, dim_feedforward=feed_forward_size,batch_first=True), num_layers=n_layers)
    
    def forward(self,src):
        return self.transformer_encoder(src)

"""
Composite model
"""
class CONSENT(nn.Module):
    def __init__(self, model1, model2, model3,sequence_size):
        super(CONSENT, self).__init__()
        self.model1 = model1
        self.model2 = model2
        self.model3 = model3
        self.sequence_size = sequence_size


    def forward(self, input_data):
        #pos_embeddings = supp_list
        input_data = input_data.view(-1, 3,128,96)
        output1 = self.model1(input_data)
         
       
        output1 = output1.view(input_data.size(0)//self.sequence_size,self.sequence_size,output1.size(-1))
        combined_output = output1 
        output2 = self.model2(combined_output)
        output2 = output2.view(-1,output2.size(-1))
        
        final_output = self.model3(output2)
        return final_output,torch.zeros((final_output.size(0),4)),torch.zeros((final_output.size(0),6))
        #return torch.zeros((final_output.size(0),4)),final_output,torch.zeros((final_output.size(0),6))

        

"""
End
"""

model2 = Encoder(emsize=embed_dim,nhead=n_heads,n_layers=num_layers,feed_forward_size=fc_size)
model1 = feature_extractor
model3 = feed_forward


model = functools.partial(CONSENT,model1=model1,model2=model2,model3=model3)
model = model(sequence_size=125)