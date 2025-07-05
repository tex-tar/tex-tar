"""
Calculate the classification report after aggregating the prediction from different context windows
 """
import torch
from torch.utils.data import DataLoader
from torchvision import transforms
import numpy as np
import json 
import os
from tqdm import tqdm
import torch.nn.functional as F
from src.dataloader.transform import loader_transform_test
from copy import deepcopy
from sklearn.metrics import classification_report

def evaluate_textar(preds_all, imgs_all, bbox_info_labels ,root_path="",sequence_size=125):
    t1 = {0:"no_bi",1:"bold",2:"italic",3:"b+i"}
    t2 = {0:"no_us",1:"underlined",2:"strikeout",3:"u+s"}
    word_bb_map = {}
    print(torch.bincount(torch.argmax(torch.tensor(preds_all[1]),dim=-1)))

    for ind,img_file in enumerate(imgs_all):
        prediction_t1 = preds_all[0][ind]
        prediction_t2 = preds_all[1][ind]

        bbid = img_file.split('_CW_')[0].split('_')[-1]
        key = '_'.join(img_file.split('_CW_')[0].split('_')[1:-1])

        try:
            word_bb_map[key+','+str(bbid)]["t1_prob"]+=F.softmax(torch.tensor(prediction_t1))
            word_bb_map[key+','+str(bbid)]["t2_prob"]+=F.softmax(torch.tensor(prediction_t2))
            word_bb_map[key+','+str(bbid)]["cnt"]+=1
        except:
            word_bb_map[key+','+str(bbid)] = {}
            word_bb_map[key+','+str(bbid)]["t1_prob"]=F.softmax(torch.tensor(prediction_t1))
            word_bb_map[key+','+str(bbid)]["t2_prob"]=F.softmax(torch.tensor(prediction_t2))
            word_bb_map[key+','+str(bbid)]["cnt"]=1
    
    bbox_info = deepcopy(bbox_info_labels)
    for key in bbox_info_labels:
        for j in range(len(bbox_info_labels[key])):
            bbox_info[key][j]["bb_ids"][0]["attb"]={
                'bold':False,
                'italic':False,
                'b+i':False,
                'no_bi': False,
                'no_us':False,
                'underlined':False,
                'strikeout':False,
                'u+s': False,
            }

    ct=0
    collection_of_keys = set()
    for key in list(word_bb_map.keys()):
        img_name = key.split(',')[0]
        collection_of_keys.add(img_name)
        bb_id = key.split(',')[1]
        pred_t1 = torch.argmax(word_bb_map[key]["t1_prob"])
        pred_t2 = torch.argmax(word_bb_map[key]["t2_prob"])
        ct+=1
        
        bbox_info[img_name][int(bb_id)]["bb_ids"][0]["attb"][t1[pred_t1.item()]]=True
        bbox_info[img_name][int(bb_id)]["bb_ids"][0]["attb"][t2[pred_t2.item()]]=True
    
    """
    Compare both the jsons
    """
    final_preds= {"t1":[],"t2":[]}
    final_labels = {"t1":[],"t2":[]}
    for key in bbox_info_labels:
        # the doc image should have >= sequence_size bboxes. 
        if len(bbox_info_labels[key])<sequence_size:
            continue
        
        for j in range(len(bbox_info_labels[key])):
            for k,attb in t1.items():
                if bbox_info[key][j]["bb_ids"][0]["attb"][attb]==True:
                        final_preds["t1"].append(k)
                if bbox_info_labels[key][j]["bb_ids"][0]["attb"][attb]==True:
                        final_labels["t1"].append(k)
            
            for k,attb in t2.items():
                if bbox_info[key][j]["bb_ids"][0]["attb"][attb]==True:
                        final_preds["t2"].append(k)
                if bbox_info_labels[key][j]["bb_ids"][0]["attb"][attb]==True:
                        final_labels["t2"].append(k)
    
    report_dict = {}
    macro_avg = 0.0
    for category in final_labels:
        report_dict[category] = classification_report(final_labels[category], final_preds[category], output_dict=True)
        print(f"Evaluate :: TexTAR overall :: F1 for {category} : {report_dict[category]['macro avg']['f1-score']}")
        macro_avg += report_dict[category]['macro avg']['f1-score']/len(final_labels)
    return report_dict,macro_avg
