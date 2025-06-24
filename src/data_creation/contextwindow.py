import os
import json
import yaml
import random
import numpy as np
import cv2

from typing import List, Dict
from utils.data_utils import crop_frame_around_points
# fix seeds
random.seed(42)
np.random.seed(42)

class ContextWindowGenerator:
    def __init__(self, config: Dict):
        """
        Initialize with parameters from config dict.
        """
        self.image_folder = config['image_folder']
        self.output_file_path = config['output_file_path']
        self.enable_random = config.get('enable_random', 0)
        self.seq_size = config['seq_size']
        self.cw_type = config['type']
        self.pad = config.get('pad', 'y').lower() == 'y'
        self.aspect_ratio = config.get('aspect_ratio', 1.3334)
        bbjson_path = config['bbjson']
        with open(bbjson_path, 'r') as f:
            self.bbox_data: Dict[str, List[dict]] = json.load(f)
        self.crops_json: Dict[str, List[dict]] = {}
        self.global_crop_count = 0

    def _prepare_choices(self, num_boxes: int) -> np.ndarray:
        """
        Initialize choice array [0,1,...,num_boxes-1].
        """
        return np.arange(num_boxes, dtype=np.int32)

    def _make_context_windows_for_image(
        self,
        image_file: str,
        bbox_array: np.ndarray,
        bbox_centres: np.ndarray,
        choices: np.ndarray
    ) -> List[Dict]:
      
        crops = []
        img = cv2.imread(os.path.join(self.image_folder, image_file))

        while np.any(choices != -1):
            try:
                visited, _, seq_crop, _ = crop_frame_around_points(
                    choice_set=choices,
                    data=bbox_centres,
                    bbox_array=bbox_array,
                    image=img,
                    args=self
                )
                # mark visited
                choices[visited] = -1

                # check sequence size
                total_tokens = sum(len(v) for v in seq_crop.values())
                if total_tokens != self.seq_size:
                    raise ValueError(
                        f"{image_file}: expected seq_size={self.seq_size}, got {total_tokens}"
                    )

                # record token counts per window
                crop_record = {str(k): len(v) for k, v in seq_crop.items()}
                crops.append(crop_record)
                self.global_crop_count += 1
            except Exception:
                break
        return crops

    def _process_single_image(self, image_file: str) -> None:
        """
        Handle crop generation (deterministic + random) for one image.
        """
        bboxes = self.bbox_data.get(image_file, [])
        if not bboxes:
            return

        dims = [b['bb_dim'] for b in bboxes]
        centres = np.array([(d[1]+d[3])//2, (d[0]+d[2])//2] for d in dims)
        bbox_array = np.array(dims)

        choices = self._prepare_choices(len(dims))
        crops1 = self._make_context_windows_for_image(
            image_file, bbox_array, np.array(centres), choices.copy()
        )

        crops2 = []
        if self.enable_random > 0:
            choices_rand = self._prepare_choices(len(dims))
            for idx, b in enumerate(bboxes):
                if b['bb_ids'][0]['attb'].get('no_bi', False):
                    choices_rand[idx] = -1

            selected = 0
            for idx in range(len(choices_rand)):
                if choices_rand[idx] == -1:
                    if random.choice([0,1,2,3]) == 0:
                        choices_rand[idx] = idx
                        selected += 1
                if selected >= self.enable_random:
                    break

            crops2 = self._make_context_windows_for_image(
                image_file, bbox_array, np.array(centres), choices_rand
            )

        self.crops_json[image_file] = crops1 + crops2

    def generate(self) -> None:
        """
        Run crop generation over all images and write output JSON.
        """
        for image_file in sorted(self.bbox_data.keys()):
            self._process_single_image(image_file)

        os.makedirs(os.path.dirname(self.output_file_path), exist_ok=True)
        with open(self.output_file_path, 'w') as fout:
            json.dump(self.crops_json, fout, indent=2)

        print(f"Generated {self.global_crop_count} crops across {len(self.crops_json)} images.")
  

def main():
    config_path = "config.yaml"
    with open(config_path, 'r') as f:
        cfg = yaml.safe_load(f)
    generator = ContextWindowGenerator(cfg)
    generator.generate()

if __name__ == '__main__':
    main()
