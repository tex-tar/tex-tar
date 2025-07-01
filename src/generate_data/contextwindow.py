import os
import json
import random
import numpy as np
import cv2
import yaml
from typing import List, Dict, Any
from src.utils.data_utils import crop_frame_around_points


class ContextCropGenerator:
    """
    Generates context-window crops for images using the entire config dict.
    Expects the config dict to have a 'context_crop' section with:
      - bbjson: path to bbox JSON
      - image_folder: folder of input images
      - output_file_path: output combined JSON
      - enable_random: number of random windows per image
      - seq_size: sequence size
      - type: 'word' or 'patch'
      - pad: 'y' or 'n'
      - aspect_ratio: float
      - offset: float
    """
    def __init__(self, config: Dict[str, Any]):
        cc = config['context_crop']
        self.bbox_json_path   = cc['bbjson']
        self.image_folder     = cc['image_folder']
        self.output_json_path = cc['output_file_path']
        self.enable_random    = cc.get('enable_random', 0)
        self.seq_size         = cc['seq_size']
        self.cw_type          = cc['type']
        self.pad              = (cc.get('pad', 'y').lower() == 'y')
        self.aspect_ratio     = cc.get('aspect_ratio', 1.3334)
        self.offset           = cc.get('offset', 0.1)

        random.seed(42)
        np.random.seed(42)

       
        if not os.path.isfile(self.bbox_json_path):
            raise FileNotFoundError(f"BBox JSON not found: {self.bbox_json_path}")
        with open(self.bbox_json_path, 'r') as f:
            self.bbox_data = json.load(f)

        self.crops_json: Dict[str, List[Dict[str, int]]] = {}
        self.total_windows = 0

    def _get_image_list(self) -> List[str]:
        exts = {'.png', '.jpg', '.jpeg', '.tiff', '.bmp'}
        return [f for f in sorted(os.listdir(self.image_folder))
                if os.path.splitext(f)[1].lower() in exts]

    def _process_single(self, image_file: str):
        boxes = self.bbox_data.get(image_file, [])
        dims = [b['bb_dim'] for b in boxes]
        centres = [((d[1] + d[3]) // 2, (d[0] + d[2]) // 2) for d in dims]
        bbox_array = np.array(dims)
        centre_array = np.array(centres)
        img = cv2.imread(os.path.join(self.image_folder, image_file))

        # deterministic windows
        choices = np.arange(len(dims), dtype=np.int32)
        windows = []
        while choices[choices != -1].size > 0:
            result = crop_frame_around_points(
                choice_set=choices,
                data=centre_array,
                bbox_array=bbox_array,
                image=img,
                args=type('A', (), {
                    'seq_size': self.seq_size,
                    'pad': 'y' if self.pad else 'n',
                    'aspect_ratio': self.aspect_ratio,
                    'type': self.cw_type
                })()
            )
            if not isinstance(result, tuple):
                break
            visited, _, seq_crop, _ = result
            choices[visited] = -1
            windows.append({str(k): len(v) for k, v in seq_crop.items()})
            self.total_windows += 1
        self.crops_json[image_file] = windows

    def run(self):
        os.makedirs(os.path.dirname(self.output_json_path), exist_ok=True)
        for img in self._get_image_list():
            print(f"Processing {img}...")
            self._process_single(img)
        with open(self.output_json_path, 'w') as f:
            json.dump(self.crops_json, f, indent=2)
        print(f"Saved {self.total_windows} context windows to {self.output_json_path}")


def main():
    # load entire config
    cfg_path = os.path.join('config', 'data_config.yaml')
    with open(cfg_path, 'r') as f:
        config = yaml.safe_load(f)
    extractor = ContextCropGenerator(config)
    extractor.run()

if __name__ == '__main__':
    main()