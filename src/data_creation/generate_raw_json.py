# generate_json.py
import os
import sys
import json
import yaml
import torch
from collections import OrderedDict
from typing import List
from doctr.io import DocumentFile
from doctr.models import ocr_predictor

class DoctrProcessor:
    def __init__(self,image_paths: List[str],output_json_dir: str, load_fine_tune_model: bool = False,pretrained_model_path: str = None,batch_size: int = 64,device: str = 'cuda:0'):
        """
        Args:
            image_paths: List of image file paths.
            output_json_dir: Directory where JSON outputs will be saved.
            load_fine_tune_model: Whether to load fine-tuned weights.
            pretrained_model_path: Path to the fine-tuned model state dict.
            batch_size: Number of images per batch.
            device: Torch device string.
        """
        self.image_paths = image_paths
        self.output_json_dir = output_json_dir
        self.load_fine_tune_model = load_fine_tune_model
        self.pretrained_model_path = pretrained_model_path
        self.batch_size = batch_size
        os.environ['USE_TORCH'] = '1'
        self.device = torch.device(device if torch.cuda.is_available() else 'cpu')
        print(f"DoctrProcessor using device: {self.device}")
        self.predictor = ocr_predictor(det_arch='db_resnet50', pretrained=True).to(self.device)
        if self.load_fine_tune_model:
            self._load_fine_tuned_model()
        if not isinstance(self.image_paths, list):
            raise TypeError("image_paths must be a list of strings.")
        for img_path in self.image_paths:
            if not os.path.isfile(img_path):
                print(f"Warning: Image file not found: {img_path}. It will be skipped.", file=sys.stderr)
        os.makedirs(self.output_json_dir, exist_ok=True)

    def _load_fine_tuned_model(self):
       
        if self.pretrained_model_path and os.path.exists(self.pretrained_model_path):
            print(f"Loading fine-tuned model from: {self.pretrained_model_path}")
            state_dict = torch.load(self.pretrained_model_path, map_location=self.device)
            new_state_dict = OrderedDict()
            for k, v in state_dict.items():
                name = k[7:]  # strip 'module.' prefix
                new_state_dict[name] = v
            try:
                self.predictor.det_predictor.model.load_state_dict(new_state_dict)
                print("Fine-tuned model loaded successfully.")
            except Exception as e:
                print(f"Error loading fine-tuned model state dict: {e}. Proceeding with default pretrained model.")
                self.load_fine_tune_model = False
        else:
            print(f"Warning: pretrained_model_path not set or file not found: {self.pretrained_model_path}. Proceeding with default pretrained model.")
            self.load_fine_tune_model = False


    def _get_doctr_predictions_for_image(self, image_path: str):
      
        doc = DocumentFile.from_images(image_path)
        result = self.predictor(doc)
        export = result.export()
        dims = [p['dimensions'] for p in export['pages']]

        words_per_page = [
            [w for blk in p['blocks'] for ln in blk['lines'] for w in ln['words']]
            for p in export['pages']
        ]

        abs_coords = []
        ocr_values = []

        for words, (h, w) in zip(words_per_page, dims):
            page_coords = []
            page_texts = []
            for word in words:
                y0, x0 = word['geometry'][0]
                y1, x1 = word['geometry'][1]
                x0i = int(round(x0 * w))
                y0i = int(round(y0 * h))
                x1i = int(round(x1 * w))
                y1i = int(round(y1 * h))
                page_coords.append([x0i, y0i, x1i, y1i])
                page_texts.append(word['value'])
            abs_coords.append(page_coords)
            ocr_values.append(page_texts)

        return ocr_values, abs_coords

    def process_images(self):
        """
        Processes images in batches and writes per-image JSON files.
        """
        total = len(self.image_paths)
        batches = (total + self.batch_size - 1) // self.batch_size
        bbox_counter = 0
        processed = 0

        print(f"Starting OCR on {total} images in {batches} batch(es).")

        import time
        start_time = time.time()

        for bi, start in enumerate(range(0, total, self.batch_size), 1):
            batch_paths = self.image_paths[start:start + self.batch_size]
            valid = [p for p in batch_paths if os.path.isfile(p)]
            if not valid:
                continue
            print(f"\n[Batch {bi}/{batches}] {len(valid)} files")

            try:
                texts, preds = self._get_doctr_predictions_for_image(valid)
                for idx, img in enumerate(valid):
                    fname = os.path.splitext(os.path.basename(img))[0]
                    out_file = os.path.join(self.output_json_dir, f"{fname}.json")
                    entries = []
                    for i, box in enumerate(preds[idx]):
                        x0, y0, x1, y1 = box
                        if x1 > x0 and y1 > y0:
                            entries.append({
                                'bb_dim': [x0, y0, x1, y1],
                                'bb_ids': [{
                                    'id': bbox_counter,
                                    'ocrv': texts[idx][i],
                                    'attb': {
                                        'no_bi': True, 'bold': False, 'italic': False, 'b+i': False,
                                        'no_us': True, 'underlined': False, 'strikeout': False, 'u+s': False
                                    }
                                }]
                            })
                            bbox_counter += 1
                    with open(out_file, 'w') as f:
                        json.dump({fname: entries}, f, indent=2)
                    print(f"  Saved: {out_file}")
                    processed += 1
            except Exception as e:
                print(f"  Error processing batch {bi}: {e}", file=sys.stderr)

            print(f"Batch time: {time.time() - start_time:.2f}s")

        print(f"Finished: {processed}/{total} images processed.")


def main():
    config_path = "config.yaml"
    with open(config_path, 'r') as f:
        cfg = yaml.safe_load(f)
    img_dir = cfg.get('image_folder')
    if not img_dir or not os.path.isdir(img_dir):
        raise ValueError(f"Invalid image_folder: {img_dir}")
    exts = cfg.get('extensions', ['.png', '.jpg', '.jpeg', '.tiff', '.bmp'])
    images = [os.path.join(img_dir, f) for f in sorted(os.listdir(img_dir))
              if os.path.splitext(f)[1].lower() in exts]
    output_json_dir=cfg['output_json_dir']
    load_fine_tune_model=cfg.get('load_fine_tune_model', False)
    pretrained_model_path=cfg.get('pretrained_model_path')
    batch_size=cfg.get('batch_size', 64)
    device=cfg.get('device', 'cuda:0')
    processor = DoctrProcessor(images,output_json_dir,load_fine_tune_model,pretrained_model_path,batch_size,device)
    processor.process_images()


if __name__ == '__main__':
    main()
