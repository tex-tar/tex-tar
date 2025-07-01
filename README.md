<div align="center">

<samp>
<h1> TexTAR  </h1>
<h2> Textual Attribute Recognition in Multi-domain and Multi-lingual Document Images </h2>
</samp>

**Accepted at ICDAR 2025**

| **[ [```Paper```](https://your-paper-link.example.com) ]** | **[ [```Website```](https://your-project-website.example.com) ]** |
|:-----------------------------------------------------------:|:---------------------------------------------------------------:|

</div>

---

## Table of Contents

1. [Getting Started](#getting-started)  
2. [Project Overview](#project-overview)  
3. [Data Preparation](#data-preparation)  
4. [Training](#training)  
7. [Evaluation & Inference](#evaluation--inference)  
8. [Fine-Tuning](#fine-tuning)  
9. [Citation](#citation)  
10. [Contact](#contact)  

---

# Getting Started

```bash
To make the code run, install the necessary libraries 
python3 -m venv .textar
pip install -r requirements.txt
```

# Project Overview


## 1. Data Creation Pipeline

1. **Bounding-Box Prediction**  
   We first run a text detector (e.g. Doctr) over each page image to produce word-level bounding boxes (`bbox.json`).

2. **Context-Window Generation**  
   For each detected box, we assemble a fixed-length context window by grouping its nearest neighbors in 2D space. This ensures that every crop carries both the target word and a consistent surrounding “context” of exactly *N* tokens.

These patches are then ready for model ingestion.

<div align="center">
  <img 
    src="assets/data-selection-pipeline.png" 
    alt="Data Selection Pipeline" 
    style="max-width: 60%; height: auto;" 
  
  />
</div>


# 2. Training Strategy

We use a **two-stage** training scheme:

### Stage 1: Base Pre-Training
- **Goal:** Learn core visual and token features.  
- **Train:** Feature Extraction Network, Transformer Encoder, Classification Heads.  
- **Outcome:** Token embeddings (`Temb`) with strong appearance cues.



### Stage 2: RoPE-MixAB Fine-Tuning
- **Goal:** Add precise positional encoding.  
- **Freeze:** Base network from Stage 1.  
- **Train:** RoPE-MixAB module + Classification Heads.  

<div align="center">
  <img 
    src="assets/model.png" 
    alt="Data Selection Pipeline" 
    style="max-width: 60%; height: auto;" 
  
  />
</div>

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

