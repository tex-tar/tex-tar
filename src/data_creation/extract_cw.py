import os
import sys
import json
import math
import cv2
import yaml
import numpy as np
from typing import List, Dict
from utils.data_utils import n_get_blocks
# fix seeds
np.random.seed(42)

def load_config(path: str = "config.yaml") -> Dict:
    with open(path, 'r') as f:
        return yaml.safe_load(f)

class ContextCropExtractor:
    """
    Extracts and saves word- or patch-level crops based on
    bounding-box JSON and context-window JSON.
    """
    def __init__(self, config: Dict):
        # basic I/O
        self.image_folder = config['image_folder']
        self.bbox_json_paths = config['bbox_json_paths']       # list of JSON files
        self.cw_json_path = config['cw_json_path']
        self.output_word_crops_path = config['output_word_crops_path']
        self.pad_cropped_frame = (config.get('pad_cropped_frame', 'y').lower() == 'y')
        self.context_window_type = config.get('context_window_type', 'word CW')
        self.sequence_size = config.get('sequence_size', 100)
        self.aspect_ratio = config.get('aspect_ratio', 1.334)
        self.word_bbox_padding_offset = config.get('word_bbox_padding_offset', 0.1)
        attrs = config.get('extract_attributes', {})
        self.extract_t1 = attrs.get('T1_bold_italic', False)
        self.extract_t2 = attrs.get('T2_underline_strikeout', False)
        self.bbox_data = self._load_bbox_jsons(self.bbox_json_paths)
        with open(self.cw_json_path, 'r') as f:
            self.cw_json = json.load(f)

        os.makedirs(self.output_word_crops_path, exist_ok=True)
        self.current_cw_offset = 0
        print(f"[Init] crops will be saved to: {self.output_word_crops_path}")

    def _load_bbox_jsons(self, paths: List[str]) -> Dict[str, List[dict]]:
      
        data: Dict[str, List[dict]] = {}
        for p in paths:
            if not os.path.exists(p):
                print(f"Error: bbox JSON not found: {p}", file=sys.stderr)
                sys.exit(1)
            with open(p, 'r') as f:
                data.update(json.load(f))
        return data

    def _load_image(self, path: str):
        img = cv2.imread(path)
        if img is None:
            print(f"Warning: failed to load image {path}", file=sys.stderr)
        return img

    def _process_single_image(self, img_path: str) -> None:
       
        fname = os.path.basename(img_path)
        if fname not in self.bbox_data or fname not in self.cw_json:
            print(f"Skipping {fname}: missing bbox or CW data")
            return
        try:
            image = cv2.imread(img_path)
            if image is None:
                return 
        except Exception as e:
            return None
        image = self._load_image(img_path)
        cw_list = self.cw_json[fname]
        bboxes   = self.bbox_data[fname]

        for local_idx, cw in enumerate(cw_list):
            cw_id = self.current_cw_offset + local_idx
            for bbid_str, expected_count in cw.items():
                bbid = int(bbid_str)
                if bbid >= len(bboxes):
                    print(f"  Warning: bbid {bbid} ≥ {len(bboxes)} for {fname}")
                    continue

                x0,y0,x1,y1 = bboxes[bbid]['bb_dim']
                if self.pad_cropped_frame:
                    pad_px = math.ceil(self.word_bbox_padding_offset * (y1 - y0))
                    y1 += pad_px

                crop = image[y0:y1, x0:x1]
                if crop.size == 0:
                    print(f"  Warning: empty crop for bbid {bbid} in {fname}")
                    continue
                blocks = n_get_blocks(crop,-1, [],original_dim=[x0,y0,x1,y1],context_window_type=self.context_window_type,aspect_ratio=self.aspect_ratio,seq_size=self.sequence_size)

                if len(blocks) != expected_count:
                    print(f"Mismatch CW#{cw_id} bbid {bbid}: expected {expected_count}, got {len(blocks)}")
                    print("error")
                    blocks = blocks[:expected_count]
                for i, blk in enumerate(blocks):
                    prefix = ""
                    attb = bboxes[bbid]['bb_ids'][0]['attb']
                    if self.extract_t1:
                        if attb.get('b+i'):   prefix += "3-"
                        elif attb.get('bold'): prefix += "1-"
                        elif attb.get('italic'): prefix += "2-"
                        elif attb.get('no_bi'): prefix += "0-"
                    if self.extract_t2:
                        if attb.get('u+s'):    prefix += "3-"
                        elif attb.get('underlined'): prefix += "1-"
                        elif attb.get('strikeout'):  prefix += "2-"
                        elif attb.get('no_us'):      prefix += "0-"
                    if not prefix:
                        prefix = "9-"

                    out_name = f"{prefix[:-1]}_{fname}_{bbid}_CW{cw_id}_{i}.png"
                    out_path = os.path.join(self.output_word_crops_path, out_name)
                    cv2.imwrite(out_path, blk)
            self.current_cw_offset += len(cw_list)

    def run(self):
        """Process every image in the folder and extract all crops."""
        all_images = sorted([
            os.path.join(self.image_folder, f)
            for f in os.listdir(self.image_folder)
            if f.lower().endswith(('.png','.jpg','.jpeg','.tiff','.bmp'))
        ])
        print(f"Processing {len(all_images)} images...")
        for img in all_images:
            self._process_single_image(img)
        print("Done.")


def main():
    cfg = load_config()["context_crop"]  
    extractor = ContextCropExtractor(cfg)
    extractor.run()


if __name__ == "__main__":
    main()