print("--- TEST_SCRIPT_ORIGINAL.PY IS BEING EXECUTED ---") 
import torch
from torchvision.transforms import ToTensor, Resize, Compose
import numpy as np
import json
import os
from PIL import Image, ImageOps
import torch.nn.functional as F
from torch.utils.data import DataLoader
from pathlib import Path 
from dataset_util import WordData, cwData

from model_files.TexTAR_rope_selective import model


class InferenceProcessor:
    def __init__(self, config_obj):
        self.config = config_obj
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.model = model.to(self.device)

        self._load_model_state()

        self.compose_transform = Compose([
            Resize((128, 96)),
            ToTensor()
        ])

        self.t1_labels = {0: "no_bi", 1: "bold", 2: "italic", 3: "b+i"}
        self.t2_labels = {0: "no_us", 1: "underlined", 2: "strikeout", 3: "u+s"}

        


    def _load_model_state(self):
        """Loads the pretrained model state."""
        pretrained_model_path = self.config.get('model_inference_config.pretrained_model')
        print("DEBUG type of pretrained_model_path:", type(pretrained_model_path))
        print(f"DEBUG: Attempting to load model from: {pretrained_model_path}")
        if not pretrained_model_path or not os.path.exists(pretrained_model_path):
            raise FileNotFoundError(f"Pretrained model not found at {pretrained_model_path}. Please check config.")

        checkpoint = torch.load(pretrained_model_path, map_location=self.device)
        self.model.load_state_dict(checkpoint['model_state_dict'])
        print("Checkpoint stats : ")
        print("train_loss : ", checkpoint.get('train_loss', 'N/A'))
        print("val_loss : ", checkpoint.get('val_loss', 'N/A'))


    def run_inference(self):
        """Runs the inference process."""
        test_type = self.config.get('model_inference_config.testTYPE')
        img_dir = self.config.get('model_inference_config.img_dir')
        all_crops_json_path = self.config.get('context_window_cropping_config.all_crops_output_json_path')
        cropped_cw_images_folder = self.config.get('context_window_cropping_config.final_cropped_images_output_folder')
        seq_size = self.config.get('model_inference_config.seq_size')

        if test_type == "patch":
            dataset = cwData(
                img_dir,
                transform=self.compose_transform,
                all_crops_json_path=all_crops_json_path,
                cropped_cw_images_folder=cropped_cw_images_folder
            )
            test_loader = DataLoader(dataset, batch_size=1, shuffle=False)
        elif test_type == "word":
            print("Running inference for word-level images.")
            dataset = WordData(img_dir, transform=self.compose_transform)
            test_loader = DataLoader(dataset, batch_size=32, shuffle=False, pin_memory=True, num_workers=4)
        else:
            raise ValueError(f"Invalid testTYPE: {test_type}. Must be 'patch' or 'word'.")

        global_stack_tensor_t1 = torch.tensor([])
        global_stack_tensor_t2 = torch.tensor([])
        global_stack_tensor_t3 = torch.tensor([])
        img_file_names_processed = [] 

        self.model.eval()
        from tqdm import tqdm
        with torch.no_grad():
            for batch_idx, batch_data in enumerate(tqdm(test_loader, desc="Inference")):
                if self.config.get('model_inference_config.testTYPE') == "patch":
                    if len(batch_data) == 3:
                        images, filenames, suppl_list = batch_data
                    elif len(batch_data) == 2: 
                        images, filenames = batch_data
                        suppl_list = None
                    else:
                        raise ValueError(f"Unexpected number of items in batch_data for patch type: {len(batch_data)}")

                    images = images.to(self.device)
                    if suppl_list is not None:
                        suppl_list = suppl_list.to(self.device) 
                    if filenames and isinstance(filenames[0], list): 
                         img_file_names_processed.extend(filenames[0])
                    else:
                         img_file_names_processed.extend(filenames) 
                else: 
                    if len(batch_data) == 2:
                        images, filenames = batch_data
                    elif len(batch_data) == 1 and isinstance(batch_data[0], tuple) and len(batch_data[0]) == 2:
                        images, filenames = batch_data[0]
                    else:
                        raise ValueError(f"Unexpected number of items in batch_data for word type: {len(batch_data)}")

                    images = images.to(self.device)
                    img_file_names_processed.extend(filenames) 

                if suppl_list is not None:
                    t1_out, t2_out, t3_out = self.model(images, suppl_list)
                else:
                    t1_out, t2_out, t3_out = self.model(images)

                global_stack_tensor_t1 = torch.cat([global_stack_tensor_t1, t1_out.cpu()])
                global_stack_tensor_t2 = torch.cat([global_stack_tensor_t2, t2_out.cpu()])
                global_stack_tensor_t3 = torch.cat([global_stack_tensor_t3, t3_out.cpu()])


        print("Shape of global_stack_tensor_t1:", global_stack_tensor_t1.shape)
        print("Shape of global_stack_tensor_t2:", global_stack_tensor_t2.shape)

        print("Number of processed image files:", len(img_file_names_processed))

        word_bb_map = self._process_predictions(img_file_names_processed, global_stack_tensor_t1,
                                                global_stack_tensor_t2, global_stack_tensor_t3, test_type)

        self._update_bbjson(word_bb_map)

    def _process_predictions(self, img_file_names, t1_preds, t2_preds, t3_preds, test_type):
        """Processes raw predictions and maps them to bounding box IDs."""
        word_bb_map = {}
        for idx, img_path_in_json in enumerate(img_file_names):
            prediction_t1 = t1_preds[idx]
            prediction_t2 = t2_preds[idx]


            img_key_for_bbjson = None
            bbid = None

            if test_type == "patch":
                parts = Path(img_path_in_json).parts
                if len(parts) == 2:
                    original_image_folder = parts[0] 
                    cropped_image_name = parts[1]    

                    img_key_for_bbjson = original_image_folder + '.jpg' #
                    bbid_part = cropped_image_name.split('_CW_')
                    if len(bbid_part) > 1:
                        bbid = bbid_part[1].split('.')[0] 
                        if not bbid.isdigit():
                            print(f"Warning: Non-digit bbid found in {cropped_image_name}: {bbid}. Setting to 0.")
                            bbid = '0'
                    else:
                        print(f"Warning: '_CW_' not found in {cropped_image_name}. Assuming bbid is 0.")
                        bbid = '0'
                else:
                    print(f"Warning: Unexpected path structure for patch: {img_path_in_json}. Skipping.")
                    continue
            else: 
                filename = Path(img_path_in_json).name 
                name_parts = filename.rsplit('_', 1)
                if len(name_parts) == 2:
                    potential_original_name_part = name_parts[0]
                    potential_bbid_part = name_parts[1].split('.')[0] 
                    
                    if potential_bbid_part.isdigit():
                        bbid = potential_bbid_part
                        img_key_for_bbjson = potential_original_name_part + '.jpg'
                    else:
                        img_key_for_bbjson = filename 
                        bbid = '0' 
                else:
                    img_key_for_bbjson = filename
                    bbid = '0' 

            if img_key_for_bbjson is None or bbid is None:
                print(f"Error: Could not parse img_key_for_bbjson or bbid for {img_path_in_json}. Skipping.")
                continue

            map_key = f"{img_key_for_bbjson},{bbid}"
            print(f"DEBUG: Processed map_key: '{map_key}' (Type: {type(map_key)})") 

            if map_key not in word_bb_map:
                word_bb_map[map_key] = {
                    "t1_prob": F.softmax(prediction_t1, dim=-1).cpu(),
                    "t2_prob": F.softmax(prediction_t2, dim=-1).cpu(),
                   
                    "cnt": 1
                }
            else:
                word_bb_map[map_key]["t1_prob"] += F.softmax(prediction_t1, dim=-1).cpu()
                word_bb_map[map_key]["t2_prob"] += F.softmax(prediction_t2, dim=-1).cpu()
                word_bb_map[map_key]["cnt"] += 1
        return word_bb_map



    def _update_bbjson(self, word_bb_map):
        """Updates the bounding box JSON with predicted attributes."""
        bbjson_path_raw = self.config.get('model_inference_config.bbjson')
        output_bbjson_path_raw = self.config.get('model_inference_config.output_bbjson')

        print(f"DEBUG_CRITICAL: bbjson_path_raw: '{bbjson_path_raw}', Type: {type(bbjson_path_raw)}")
        print(f"DEBUG_CRITICAL: output_bbjson_path_raw: '{output_bbjson_path_raw}', Type: {type(output_bbjson_path_raw)}")

        if isinstance(bbjson_path_raw, tuple):
            bbjson_path = bbjson_path_raw[0]
            print(f"DEBUG_CRITICAL: Corrected bbjson_path to: '{bbjson_path}' (was tuple)")
        else:
            bbjson_path = bbjson_path_raw

        if isinstance(output_bbjson_path_raw, tuple):
            output_bbjson_path = output_bbjson_path_raw[0]
            print(f"DEBUG_CRITICAL: Corrected output_bbjson_path to: '{output_bbjson_path}' (was tuple)")
        else:
            output_bbjson_path = output_bbjson_path_raw
        output_dir = os.path.dirname(output_bbjson_path)
        if output_dir and not os.path.exists(output_dir):
            os.makedirs(output_dir, exist_ok=True)
            print(f"Created output directory: {output_dir}")

        if not os.path.exists(bbjson_path):
            raise FileNotFoundError(f"Input bbjson not found at {bbjson_path}. Please check config.")

        with open(bbjson_path, 'r') as file:
            bbox_info = json.load(file)

        for img_name_key in bbox_info:
            for j in range(len(bbox_info[img_name_key])):
                if not bbox_info[img_name_key][j].get("bb_ids") or not isinstance(bbox_info[img_name_key][j]["bb_ids"], list) or len(bbox_info[img_name_key][j]["bb_ids"]) == 0:
                    bbox_info[img_name_key][j]["bb_ids"] = [{}]
                if "attb" not in bbox_info[img_name_key][j]["bb_ids"][0]:
                    bbox_info[img_name_key][j]["bb_ids"][0]["attb"] = {}

                initial_attb = {
                    'bold': False, 'italic': False, 'b+i': False, 'no_bi': False,
                    'no_us': False, 'underlined': False, 'strikeout': False, 'u+s': False,
                }
                bbox_info[img_name_key][j]["bb_ids"][0]["attb"].update(initial_attb)

        processed_bb_keys = set()
        for map_key in word_bb_map.keys():
            print(f"DEBUG: Processing map_key in _update_bbjson loop: '{map_key}' (Type: {type(map_key)})")
            img_name_for_bbjson, bb_id_str = map_key.split(',')

            print(f"DEBUG: img_name_for_bbjson: '{img_name_for_bbjson}' (Type: {type(img_name_for_bbjson)})")
            print(f"DEBUG: bb_id_str: '{bb_id_str}' (Type: {type(bb_id_str)})")

            bb_id = int(bb_id_str)

            pred_t1_prob = word_bb_map[map_key]["t1_prob"]
            pred_t2_prob = word_bb_map[map_key]["t2_prob"]
            pred_count = word_bb_map[map_key]["cnt"]

            pred_t1 = torch.argmax(pred_t1_prob / pred_count)
            pred_t2 = torch.argmax(pred_t2_prob / pred_count)
       
            if img_name_for_bbjson in bbox_info and bb_id < len(bbox_info[img_name_for_bbjson]):
                attb_dict = bbox_info[img_name_for_bbjson][bb_id]["bb_ids"][0]["attb"]

                for label_key in self.t1_labels.values():
                    attb_dict[label_key] = False
                for label_key in self.t2_labels.values():
                    attb_dict[label_key] = False

                attb_dict[self.t1_labels[pred_t1.item()]] = True
                attb_dict[self.t2_labels[pred_t2.item()]] = True
                
                processed_bb_keys.add(map_key)
            else:
                print(f"Warning: Original image key '{img_name_for_bbjson}' or bounding box ID {bb_id} not found in original bbjson. Skipping update for {map_key}.")

        print(f"Successfully processed and updated attributes for {len(processed_bb_keys)} bounding boxes.")

        with open(output_bbjson_path, 'w') as file:
            json.dump(bbox_info, file, indent=4)
        print(f"Updated bounding box information saved to {output_bbjson_path}")