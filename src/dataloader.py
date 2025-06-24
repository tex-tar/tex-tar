from torch.utils.data import DataLoader,Dataset, Subset
from torchvision.transforms import ToTensor,Resize,Compose
import os
import torch
from PIL import Image
from utils.dataloader.custom_label import labelize   
import numpy as np 
import random
# from utils.dataloader.transform import transform_italicize
import pandas as pd  
from PIL import Image, ImageFile
import json
from transformers import AutoImageProcessor
import math
#from utils.helper import pos2posemb2d
import pdb

class DocLevelDataset_RoPE_Train(Dataset):
    def __init__(self,order_list,img_dir,label_split,total_categories,transform=None):
        self.order_list = os.listdir(img_dir)
        self.img_dir = img_dir
        self.transform = transform
        self.label_split = label_split
        self.total = total_categories
        #self.bbjson_path = "/ssd_scratch/rohan/swaroopajinka/hisam_outputs/train_verified_hisam.json"
        self.bbjson_path = "/ssd_scratch/rohan.kumar/swaroopajinka/corpus_augmented_decreased_underlines/train/labels/train_1005_underline_final2.json"
        with open(self.bbjson_path,'r') as file:
            self.bbjson = json.load(file)

    def __len__(self):
        return len(self.order_list)
    
    def __getitem__(self, idx):
        images = []
        labels = []
        xs=[]
        ys=[]
        coods_list=[]

        ind = self.order_list[idx]
        
        folder_path = os.path.join(self.img_dir,str(ind))
        for image_file in os.listdir(folder_path):
            doc_name = '_'.join(image_file.split('_CW_')[0].split('_')[1:-1])
            bbid = int(image_file.split('_CW_')[0].split('_')[-1])
            coods = self.bbjson[doc_name][bbid]["bb_dim"]
            xs.append(coods[0])
            xs.append(coods[2])
            ys.append(coods[1])
            ys.append(coods[3])
        xs.sort()
        ys.sort()
        norm_x = (xs[-1]-xs[0])
        norm_y = (ys[-1]-ys[0])
            
        for image_file in os.listdir(folder_path):
            labels.append(labelize(image_file,self.label_split,self.total))
            
            img = Image.open(os.path.join(folder_path,image_file)).convert('RGB')
            
            # extract hw,coods
            doc_name = '_'.join(image_file.split('_CW_')[0].split('_')[1:-1])
            bbid = int(image_file.split('_CW_')[0].split('_')[-1])
            
            coods = self.bbjson[doc_name][bbid]["bb_dim"]
            coods_list.append(torch.tensor([((coods[0]-xs[0])/norm_x + (coods[2]-xs[0])/norm_x)/2,((coods[1]-ys[0])/norm_y+(coods[3]-ys[0])/norm_y)/2]))
            
            if self.transform:
                img = self.transform(img)
            images.append(img)
       
        images = torch.stack(images)
        labels = torch.stack(labels)
        
        coods_list = torch.stack(coods_list)
        return (images,labels,{'coods':coods_list})
        # return (images,labels)


class DocLevelDataset_RoPE_Val(Dataset):
    def __init__(self,order_list,img_dir,label_split,total_categories,transform=None):
        self.order_list = os.listdir(img_dir)
        self.img_dir = img_dir
        self.transform = transform
        self.label_split = label_split
        self.total = total_categories
        self.bbjson_path = "/ssd_scratch/rohan/swaroopajinka/corpus_augmented_decreased_underlines/val/labels/val_augmented.json"
        with open(self.bbjson_path,'r') as file:
            self.bbjson = json.load(file)

    def __len__(self):
        return len(self.order_list)
    
    
    
    def __getitem__(self, idx):
        images = []
        labels = []
        xs=[]
        ys=[]
        coods_list=[]

        ind = self.order_list[idx]
        
        folder_path = os.path.join(self.img_dir,str(ind))
        for image_file in os.listdir(folder_path):
            doc_name = '_'.join(image_file.split('_CW_')[0].split('_')[1:-1])
            bbid = int(image_file.split('_CW_')[0].split('_')[-1])
            coods = self.bbjson[doc_name][bbid]["bb_dim"]
            xs.append(coods[0])
            xs.append(coods[2])
            ys.append(coods[1])
            ys.append(coods[3])
        xs.sort()
        ys.sort()
        norm_x = (xs[-1]-xs[0])
        norm_y = (ys[-1]-ys[0])
            
        for image_file in os.listdir(folder_path):
            labels.append(labelize(image_file,self.label_split,self.total))
            
            img = Image.open(os.path.join(folder_path,image_file)).convert('RGB')
            
            # extract hw,coods
            doc_name = '_'.join(image_file.split('_CW_')[0].split('_')[1:-1])
            bbid = int(image_file.split('_CW_')[0].split('_')[-1])
            
            coods = self.bbjson[doc_name][bbid]["bb_dim"]
            coods_list.append(torch.tensor([((coods[0]-xs[0])/norm_x + (coods[2]-xs[0])/norm_x)/2,((coods[1]-ys[0])/norm_y+(coods[3]-ys[0])/norm_y)/2]))
            
            if self.transform:
                img = self.transform(img)
            images.append(img)
       
        images = torch.stack(images)
        labels = torch.stack(labels)
        
        coods_list = torch.stack(coods_list)
        return (images,labels,{'coods':coods_list})