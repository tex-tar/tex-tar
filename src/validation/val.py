import os
import yaml
import json
import time
import torch
import numpy as np
from torch.utils.data import DataLoader
from utils.dataloader.transform import loader_transform_val
from utils.helper import load_module, load_config
from utils.run_utils import calculate_weighted_average
from utils.files.metrics_added_forf1metric import calculate_f1_scores



class Validator:
    """
    Validation loop for model performance, computing loss, accuracy, and F1 metrics.

    Config keys:
      - datasets: list of dataset modules for validation
      - dataloaders: list of dataloader classes
      - order_files
      - batch_split, sequence_size, num_workers, shuffle
      - loss_fn (list of loss function modules)
      - loss_types, loss_weights, label_split, total_categories
      - bbox_info_path: path to ground-truth JSON for F1 computation
    """
    def __init__(self, cfg: dict, model: torch.nn.Module):

        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.model = model.to(self.device)
        self.loss_types = cfg['loss_types']
        self.loss_weights = cfg['loss_weights']
        self.label_split = cfg['label_split']
        self.total_categories = cfg['total_categories']
        self.sequence_size = cfg['sequence_size']
        self.loss_fns = [load_module(m, 'loss_functions').calculate_loss for m in cfg['loss_fn']]
        self.dataloaders = self._init_dataloaders(cfg)

        with open(cfg['bbox_info_path'], 'r') as f:
            self.gt_bbox_info = json.load(f)

    def _init_dataloaders(self, cfg: dict):
        loaders = []
        lengths = []
        for ds_cfg, dl_cfg in zip(cfg['datasets'], cfg['dataloaders']):
            Dataset = load_module(ds_cfg['module'], 'dataloader')
            dataset = Dataset(
                order_list=ds_cfg['order_list'],
                img_dir=ds_cfg['img_dir'],
                label_split=ds_cfg['label_split'],
                total_categories=ds_cfg['total_categories'],
                transform=loader_transform_val
            )
            loader = DataLoader(
                dataset,
                batch_size=dl_cfg['batch_size'],
                shuffle=dl_cfg['shuffle'],
                num_workers=cfg['num_workers'],
                pin_memory=True
            )
            loaders.append(loader)
            lengths.append(len(loader))
        return loaders

    def eval(self, epoch: int):
        """
        Runs validation over all dataloaders.
        Returns (avg_val_loss, t1_f1, t2_f1).
        """
        self.model.eval()
        total_loss = 0.0
        count = 0

        # accumulate predictions per bounding box
        prob_map = {}

        with torch.no_grad():
            start = time.time()
            for loader_idx, loader in enumerate(self.dataloaders):
                for images, labels, supplementary in loader:
                    # reshape and move to device
                    B, seq, C, H, W = images.shape
                    imgs = images.view(-1, C, H, W).to(self.device)
                    lbls = labels.view(-1, labels.shape[-1]).to(self.device)
                    coords = supplementary['coods'].to(self.device)
                    img_files = supplementary['img_files']  
                    t1_out, t2_out, _, _ = self.model(imgs, [coords], mode='val')
                    loss = self.loss_fns[0](
                        torch.cat([t1_out, t2_out], dim=-1),
                        lbls,
                        self.loss_types,
                        self.loss_weights,
                        self.total_categories,
                        self.sequence_size,
                        self.label_split[loader_idx]
                    )
                    total_loss += loss.item()
                    count += 1

                    # accumulate probabilities
                    probs1 = torch.softmax(t1_out, dim=-1).cpu()
                    probs2 = torch.softmax(t2_out, dim=-1).cpu()
                    flat_names = list(itertools.chain.from_iterable(img_files))
                    for i, name in enumerate(flat_names):
                        key = self._make_map_key(name)
                        entry = prob_map.setdefault(key, {'p1':0, 'p2':0, 'cnt':0})
                        entry['p1'] += probs1[i]
                        entry['p2'] += probs2[i]
                        entry['cnt'] += 1

        avg_loss = total_loss / count
        print(f"Epoch {epoch} validation loss: {avg_loss:.4f} (time: {time.time()-start:.1f}s)")

        # compute predictions per bbox
        pred_info = self._aggregate_predictions(prob_map)
        t1_f1, t2_f1 = calculate_f1_scores(self.gt_bbox_info, pred_info)
        w1, w2 = calculate_weighted_average(t1_f1, t2_f1)

        return avg_loss, w1, w2

    def _make_map_key(self, img_file: str) -> str:
        # extract "imagename,bbid" from filename
        base, _ = img_file.split('_CW_')
        parts = base.split('_')
        bbid = parts[-1]
        img = '_'.join(parts[1:-1])
        return f"{img},{bbid}"

    def _aggregate_predictions(self, prob_map: dict) -> dict:
        # average probs and assign predicted attributes
        pred = {}
        for key, v in prob_map.items():
            avg1 = v['p1'] / v['cnt']
            avg2 = v['p2'] / v['cnt']
            p1 = torch.argmax(avg1).item()
            p2 = torch.argmax(avg2).item()
            pred[key] = (p1, p2)
        return pred


