from torchvision.models import resnet18
import torch.nn as nn
import functools
import numpy as np
import torch
from functools import partial
import copy

import torch.nn.functional as F

from timm.models.vision_transformer import Mlp, PatchEmbed , _cfg

from timm.models.layers import DropPath, to_2tuple, trunc_normal_
from timm.models.registry import register_model

from src.architectures.deit.models_v2 import Layer_scale_init_Block, Attention



embed_dim = 256
num_layers = 6
n_heads = 8
rope_stacks=2

fc_size=256


def init_random_2d_freqs(dim: int, num_heads: int, theta: float = 10.0, rotate: bool = True):
    freqs_x = []
    freqs_y = []
    mag = 1 / (theta ** (torch.arange(0, dim, 4)[: (dim // 4)].float() / dim))
    for i in range(num_heads):
        angles = torch.rand(1) * 2 * torch.pi if rotate else torch.zeros(1)        
        fx = torch.cat([mag * torch.cos(angles), mag * torch.cos(torch.pi/2 + angles)], dim=-1)
        fy = torch.cat([mag * torch.sin(angles), mag * torch.sin(torch.pi/2 + angles)], dim=-1)
        freqs_x.append(fx)
        freqs_y.append(fy)
    freqs_x = torch.stack(freqs_x, dim=0)
    freqs_y = torch.stack(freqs_y, dim=0)
    freqs = torch.stack([freqs_x, freqs_y], dim=0)
    return freqs


def compute_mixed_cis(freqs: torch.Tensor, t_x: torch.Tensor, t_y: torch.Tensor, num_heads: int):
    B,N = t_x.shape
    depth = freqs.shape[1]
    # No float 16 for this range

    with torch.cuda.amp.autocast(enabled=False):
        freqs_x = (t_x.unsqueeze(-1).unsqueeze(1) @ freqs[0].unsqueeze(-2)).view(B,depth, N, num_heads, -1).permute(0, 1, 3, 2, 4)
        freqs_y = (t_y.unsqueeze(-1).unsqueeze(1) @ freqs[1].unsqueeze(-2)).view(B,depth, N, num_heads, -1).permute(0, 1, 3, 2, 4)

        freqs_cis = torch.polar(torch.ones_like(freqs_x), freqs_x + freqs_y)
    
    return freqs_cis


def init_t_xy(coods):
    t_x = coods[...,0]
    t_y = coods[...,1]
    return t_x,t_y


def apply_rotary_emb(xq: torch.Tensor, xk: torch.Tensor, freqs_cis: torch.Tensor):
    xq_ = torch.view_as_complex(xq.float().reshape(*xq.shape[:-1], -1, 2))
    xk_ = torch.view_as_complex(xk.float().reshape(*xk.shape[:-1], -1, 2))
    xq_out = torch.view_as_real(xq_ * freqs_cis).flatten(3)
    xk_out = torch.view_as_real(xk_ * freqs_cis).flatten(3)
    return xq_out.type_as(xq).to(xq.device), xk_out.type_as(xk).to(xk.device)

class RoPEAttention(Attention):
    """Multi-head Attention block with rotary position embeddings."""
    def forward(self, x, freqs_cis):
        B, N, C = x.shape
        qkv = self.qkv(x).reshape(B, N, 3, self.num_heads, C // self.num_heads).permute(2, 0, 3, 1, 4)
        q, k, v = qkv[0], qkv[1], qkv[2]
        
        q, k = apply_rotary_emb(q, k, freqs_cis=freqs_cis)
        attn = (q * self.scale) @ k.transpose(-2, -1)
        attn = attn.softmax(dim=-1)
        attn = self.attn_drop(attn)

        x = (attn @ v).transpose(1, 2).reshape(B, N, C)
        x = self.proj(x)
        x = self.proj_drop(x)
        
        return x

class Layer_scale_init_Block(nn.Module):
    # taken from https://github.com/rwightman/pytorch-image-models/blob/master/timm/models/vision_transformer.py
    # with slight modifications
    def __init__(self, dim, num_heads, mlp_ratio=4., qkv_bias=False, qk_scale=None, drop=0., attn_drop=0.,
                 drop_path=0., act_layer=nn.GELU, norm_layer=nn.LayerNorm,Attention_block = Attention,Mlp_block=Mlp
                 ,init_values=1e-4):
        super().__init__()
        self.norm1 = norm_layer(dim)
        self.attn = Attention_block(
            dim, num_heads=num_heads, qkv_bias=qkv_bias, qk_scale=qk_scale, attn_drop=attn_drop, proj_drop=drop)
        # NOTE: drop path for stochastic depth, we shall see if this is better than dropout here
        self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()
        self.norm2 = norm_layer(dim)
        mlp_hidden_dim = int(dim * mlp_ratio)
        self.mlp = Mlp_block(in_features=dim, hidden_features=mlp_hidden_dim, act_layer=act_layer, drop=drop)


class RoPE_Layer_scale_init_Block(Layer_scale_init_Block):
    # taken from https://github.com/rwightman/pytorch-image-models/blob/master/timm/models/vision_transformer.py
    # with slight modifications
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def forward(self, x, freqs_cis):
        # removed gamma1 and gamma2 
        x = x + self.drop_path(self.attn(self.norm1(x), freqs_cis=freqs_cis))
        x = x + self.drop_path(self.mlp(self.norm2(x)))
        
        return x



feature_extractor = resnet18(pretrained=True)
in_features = feature_extractor.fc.in_features
feature_extractor.fc = nn.Sequential(
    nn.Linear(in_features, 256),
    nn.ReLU()
    # nn.Linear(256, 64)
)

feed_forward_t1 = nn.Sequential(
    nn.Linear(512,128), # 64 dim embedding obtained from encoder module
    nn.Dropout(0.20),
    nn.ReLU(),
    nn.Linear(128,64),
    nn.Dropout(0.20),
    nn.ReLU(),
    nn.Linear(64,32),
    nn.Dropout(0.20),
    nn.ReLU(),
    nn.Linear(32,4)
)

feed_forward_t2 = nn.Sequential(
    nn.Linear(512,128), # 64 dim embedding obtained from encoder module
    nn.Dropout(0.20),
    nn.ReLU(),
    nn.Linear(128,64),
    nn.Dropout(0.20),
    nn.ReLU(),
    nn.Linear(64,32),
    nn.Dropout(0.20),
    nn.ReLU(),
    nn.Linear(32,4)
)


class Encoder(nn.Module):
    def __init__(self, emsize=64,nhead=2,n_layers = 4, feed_forward_size=64):
        super(Encoder, self).__init__()
        self.transformer_encoder = nn.TransformerEncoder(nn.TransformerEncoderLayer(d_model=emsize, nhead=nhead, dim_feedforward=feed_forward_size,batch_first=True,dropout=0.20), num_layers=n_layers)
    
    def forward(self,src):
        return self.transformer_encoder(src)

"""
Composite model
"""

class TexTAR(nn.Module):
    def __init__(self, model1, model2, model3,model4,sequence_size,num_heads,rope_theta, img_size,norm_weights_x,norm_weights_y):
        super(TexTAR, self).__init__()
        self.model1 = model1
        self.model2 = model2
        self.model3_head = model3
        self.num_heads = num_heads
        self.model4_head = model4
        self.blocks = nn.ModuleList([RoPE_Layer_scale_init_Block(dim=embed_dim, num_heads=num_heads, mlp_ratio=1., qkv_bias=True, qk_scale=None, drop=0., attn_drop=0.,drop_path=0., act_layer=nn.ReLU, Attention_block = RoPEAttention,Mlp_block=Mlp) for i in range(rope_stacks)])
        self.sequence_size = sequence_size
        self.rope_mixed=True
        self.norm_weights_x = norm_weights_x 
        self.norm_weights_y = norm_weights_y
        self.norm = nn.LayerNorm(embed_dim,eps=1e-06)

        if self.rope_mixed:
            self.compute_cis = partial(compute_mixed_cis, num_heads=self.num_heads)
            
            freqs = []
            for i, _ in enumerate(self.blocks):
                freqs.append(
                    init_random_2d_freqs(dim=embed_dim // num_heads, num_heads=num_heads, theta=rope_theta)
                )
            freqs = torch.stack(freqs, dim=1).view(2, len(self.blocks), -1)
            self.freqs = nn.Parameter(freqs.clone(), requires_grad=True)

    def forward(self,input_data,supp):
        coods = torch.stack(supp['coods']) if not torch.is_tensor(supp['coods']) else supp['coods']
        coods = coods.view(-1,coods.size(-1))
        coods = coods.view(coods.size(0)//self.sequence_size,self.sequence_size,coods.size(-1))
        input_data = input_data.view(-1,3,128,96)
        output1 = self.model1(input_data)
        
        output1 = output1.view(input_data.size(0)//self.sequence_size,self.sequence_size,output1.size(-1))
        output2 = self.model2(output1)

        x = copy.deepcopy(output2)
        t_x, t_y = init_t_xy(coods)
        t_x = (self.norm_weights_x * t_x)
        t_y = (self.norm_weights_y * t_y)
        t_x = t_x.cuda()
        t_y = t_y.cuda()
        freqs_cis = self.compute_cis(self.freqs, t_x, t_y)
        
        for i , blk in enumerate(self.blocks):
            x = blk(x, freqs_cis=freqs_cis[:,i])

        x = self.norm(x)
        x = x.view(-1,x.size(-1))
        output2 = output2.view(-1,output2.size(-1))
        
        output_cat = torch.cat([x,output2],dim=-1)
        final_output1 = self.model3_head(output_cat)
        final_output2 = self.model4_head(output_cat)
        
        return torch.cat([final_output1,final_output2],dim=-1)

"""
End
"""

model2 = Encoder(emsize=embed_dim,nhead=n_heads,n_layers=num_layers,feed_forward_size=fc_size)
model1 = feature_extractor
model3 = feed_forward_t1
model4 = feed_forward_t2

model = functools.partial(TexTAR,model1=model1,model2=model2,model3=model3,model4=model4,num_heads=n_heads,rope_theta=10, img_size=(128,96),norm_weights_x=60,norm_weights_y=14)
model = model(sequence_size=125)