<div align="center">

<samp>
<h1> TexTAR  </h1>
<h2> Textual Attribute Recognition in Multi-domain and Multi-lingual Document Images </h2>
</samp>

**Accepted at ICDAR 2025**

| **[ [```Paper```](https://your-paper-link.example.com) ]** | **[ [```Website```](https://tex-tar.github.io/) ]** |
|:-----------------------------------------------------------:|:---------------------------------------------------------------:|

</div>

---

## Table of Contents

1. [Getting Started](#getting-started)  
2. [Project Overview](#project-overview)   
3. [Evaluation & Inference](#evaluation--inference)  
4. [Fine-Tuning](#fine-tuning)  
5. [Citation](#citation)  
6. [Contact](#contact)  

---

# Getting Started

```bash
To make the code run, install the necessary libraries 
python3 -m venv .textar
pip install -r requirements.txt

To generate context windows, run src/generate_data/data_pipeline.py
To train a model, run src/main.py
To inference a model, run src/inference/infer.py
```

# Project Overview

TexTAR is a context-aware Transformer for Textual Attribute Recognition (TAR), handling bold, italic, underline, and strike-out across noisy, multilingual documents. We introduce a fast data-selection pipeline that builds fixed-length context windows and a 2D Rotary Positional Embedding module to fuse local context. TexTAR outperforms prior methods, showing that richer context yields superior TAR accuracy.
## 1. Data Creation Pipeline

1. **Bounding-Box Prediction**  
   We first run a text detector (e.g. Doctr) over each page image to produce word-level bounding boxes (`bbox.json`).

2. **Context-Window Generation**  
   For each detected box, we assemble a fixed-length context window by grouping its nearest neighbors in 2D space. This ensures that every crop carries both the target word and a consistent surrounding “context” of exactly *N* tokens.

These word level images are then ready for model ingestion.

<div align="center">
  <img 
    src="assets/data-selection-pipeline.png" 
    alt="Data Selection Pipeline" 
    style="max-width: 60%; height: auto;" 
  
  />
</div>


## 2. Training Strategy

We employ a **two-stage** training scheme:

**Stage 1: Base Pre-training**  
- Train the Feature Extraction Network (FEN), Transformer Encoder (TEnc) and Dual Classification Heads end-to-end on context-window crops.  
- Outcome: strong token embeddings (Temb) carrying rich visual and coarse positional cues.

**Stage 2: RoPE-MixAB Fine-tuning**  
- Freeze FEN and TEnc to preserve learned features.  
- Train the RoPE-MixAB module (and continue fine-tuning the classification heads).  
- Concatenate Temb with the RoPE-MixAB output (Trope) before the final heads, injecting precise positional information without disturbing base representations.

<div align="center">
  <img 
    src="assets/model.png" 
    alt="Model Architecture" 
    style="max-width: 60%; height: auto;" 
  
  />
</div>

# ⚙️ Configuration
- **Data Creation Pipeline**: edit `config/data_config.yaml` to adjust paths, sequence size, random crop settings, etc for extracting data to give the input to the model 
- **Model Training**: edit `config/train_config.yaml` (or `config/config.yaml`) to set learning rate, batch size, model name, datasets, optimizer, and other hyperparameters for training the model.

| Parameter                                     | Description          | Default Value                                           |
|---------------------------------------|------------------------------------------------------------------------------------------------------------------------------|---------------------------------------------------------|
| input_images                                           | Path to the folder containing source document images.                                                                        | /data/textar_inputs                                     |
| bbox_json                             | Output JSON from bounding‐box generation (maps each image to its word‐level bboxes).                                         | /data/textar_outputs/jsons/generate_bbox.json           |
| crops_json                            | Output JSON from context‐window cropping (maps each image to its context windows).                                            | /data/textar_outputs/jsons/crops.json                   |
| word_only                             | Directory where per‐word crops (after extract_cw) are saved.                                                                  | /data/textar_outputs/word_only                          |
| word_cw                               | Final directory for grouped CW folders (after image_organization).                                                            | /data/textar_outputs/word_cw                            |
| generate_json.image_folder            | Folder of raw input images for OCR/bbox predictor.                                                                           | *INPUT_IMAGES                                           |
| generate_json.output_json             | Path where bounding‐box JSON will be written.                                                                                 | *BBOX_JSON                                              |
| generate_json.batch_size              | Batch size for the OCR/bbox predictor.                                                                                        | 64                                                      |
| generate_json.device                  | Torch device for OCR (e.g. cuda:0 or cpu).                                                                                    | cuda:0                                                  |
| context_crop.bbjson                   | Path to the bbox JSON used by the context‐window generator.                                                                  | *BBOX_JSON                                              |
| context_crop.output_file_path         | Path where the context‐window definitions will be saved.                                                                     | *CROPS_JSON                                             |
| context_crop.seq_size                 | Number of tokens (words/patches) per fixed‐length context window.                                                             | 125                                                     |
| context_crop.type                     | "word" for word‐based windows or "patch" for sub‐patch windows.                                                               | word                                                    |
| extract_cw.cw_json_path               | The context‐window JSON generated in the previous step.                                                                       | *CROPS_JSON                                             |
| extract_cw.output_word_crops_path     | Directory into which each word’s final image crops will be written.                                                           | *WORD_ONLY_DIR                                          |
| image_organization.source_folder      | Source folder of all extracted word images to re‐group per CW.                                                               | *WORD_ONLY_DIR                                          |
| image_organization.dest_folder        | Destination of the final context windows.                                                           | *WORD_CW_DIR 

     
# Evaluation & Inference

- **Metric:** We use the F1-score to balance precision and recall, making it robust to the natural class imbalance in attributes like bold, italic, underline, and strikeout.  
- **Inference Pipeline:** At test time, each image goes through our two-stage pipeline (bbox detection → context-window generation → model prediction). The model’s per-word attribute outputs are compared against ground truth using the F1 metric to quantify recognition performance.                                  

# Citation
Please use the following BibTeX entry for citation .
```bibtex
@article{Kumar2025TexTAR,
  title   = {TexTAR: Textual Attribute Recognition in Multi-domain and Multi-lingual Document Images},
  author  = {Rohan Kumar and Jyothi Swaroopa Jinka and Ravi Kiran Sarvadevabhatla},
  journal = {arXiv},
  year    = {2025}
}

```
# Contact
For any queries, please contact [Dr. Ravi Kiran Sarvadevabhatla](mailto:ravi.kiran@iiit.ac.in.)

# License
This project is open sourced under [MIT License](LICENSE).

