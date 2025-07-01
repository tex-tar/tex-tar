#!/usr/bin/env python3
import os
import sys
import yaml
import json
import cv2
from typing import Any, Dict
from src.utils.data_utils import extract_cw

class ContextCWExtractor:
    
    def __init__(self, cfg):

        ec = cfg.get('extract_cw', {})
        
        self.image_folder: str = ec['image_folder']
        self.bbox_json_path: str = ec['bbox_json_path']
        self.cw_json_path: str = ec['cw_json_path']
        self.output_folder: str = ec['output_word_crops_path']
        os.makedirs(self.output_folder, exist_ok=True)

        self.bbjson: Dict[str, Any] = json.load(open(self.bbox_json_path, 'r'))
        self.cw_json: Dict[str, Any] = json.load(open(self.cw_json_path, 'r'))


        self.extract_dict: Dict[str, bool] = ec.get('extract_attributes', {})

        self.params: Dict[str, Any] = {
            'path': self.output_folder,
            'offset': ec.get('word_bbox_padding_offset', 0.1)
        }

       
        class Args:
            pass
        self.args = Args()
        # pad: 'y' or 'n'
        self.args.pad = 'y' if ec.get('pad_cropped_frame', 'y').lower() == 'y' else 'n'
        # type: 'word' or 'patch'
        cw_type = ec.get('context_window_type', 'word CW')
        self.args.type = 'word' if cw_type.lower().startswith('word') else 'patch'
        self.args.seq_size = ec.get('sequence_size', 100)
        self.args.aspect_ratio = ec.get('aspect_ratio', 1.3334)

        # Running offset across images
        self.cw_offset = 0

    def run(self) -> None:
        """
        Iterate through each image in the folder, call `extract_cw`,
        and update the running `cw_offset`.
        """
        exts = ('.png', '.jpg', '.jpeg', '.tiff', '.bmp')
        for fname in sorted(os.listdir(self.image_folder)):
            if not fname.lower().endswith(exts):
                continue
            image_path = os.path.join(self.image_folder, fname)
            image = cv2.imread(image_path)
            if image is None:
                print(f"Warning: could not load image {image_path}", file=sys.stderr)
                continue
            # call util function
            self.cw_offset = extract_cw(
                page_image=image,
                image_path=image_path,
                bbjson=self.bbjson,
                cw_json=self.cw_json,
                extract_dict=self.extract_dict,
                params=self.params,
                args=self.args,
                cw_offset=self.cw_offset
            )
        print(f"All done. Last cw_offset = {self.cw_offset}")


def main():
    # assume script is placed at project root or adjust path
    config_path = os.path.join( 'config', 'data_config.yaml')
    extractor = ContextCWExtractor(config_path)
    extractor.run()

if __name__ == '__main__':
    main()
