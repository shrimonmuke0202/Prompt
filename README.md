# Model Agnostic Graph Prompt Learning for Crystal Property Prediction
 
**Accepted at UAI 2026** 🎉

## 📄 About
 
This repository will host the official implementation of our UAI 2026 paper:
 
> **Model Agnostic Graph Prompt Learning for Crystal Property Prediction**
> Shrimon Mukherjee\*, Kishalay Das\*, Pawan Goyal, Partha Basuchowdhuri, Niloy Ganguly
> *Uncertainty in Artificial Intelligence (UAI), 2026*
>

## 🔍 Overview
 
Graph neural networks (GNNs) have become a dominant paradigm for crystal structure representation and property prediction, but adapting large pretrained crystal encoders to new downstream properties typically requires expensive full fine-tuning. This work introduces a **model-agnostic graph prompt learning** framework that adapts frozen, pretrained crystal GNN encoders to downstream property-prediction tasks by learning lightweight, task-specific graph prompts, rather than updating the full set of backbone parameters.
 
The proposed approach is designed to plug into a variety of crystal GNN backbones, offering an efficient and generalizable alternative to full fine-tuning for crystal property prediction.
 
*(A detailed technical description, architecture diagram, and results table will be added once the code and camera-ready paper are released.)*

## 📦 Repository Structure

This repository contains a reference implementation of our graph prompt learning
framework applied to the **Matformer** backbone.

```
Prompt-main/
├── LICENSE
├── requirements.txt
└── matformer/
    ├── config.py            # Training / model configuration
    ├── data.py              # Dataset loading and preprocessing
    ├── features.py          # Atom / crystal feature construction
    ├── graphs.py            # Crystal-graph construction utilities
    ├── train.py             # Core training loop
    ├── train_props.py       # Generic property-training entry (train_prop_model)
    ├── train_jv.py          # JARVIS-DFT property training entry point
    ├── train_mp.py          # Materials Project (MEGNet) training entry point
    ├── utils.py
    ├── models/
    │   ├── pyg_att.py        # Matformer backbone + graph prompt modules
    │   │                     #   (NodePrompt + graph-level prompt embedding)
    │   ├── transformer.py
    │   ├── bn_utils.py
    │   └── utils.py
    ├── mp_bulk/             # MP bulk-modulus splits (train/val/test .pkl)
    └── mp_shear/            # MP shear-modulus splits (train/val/test .pkl)
```

The graph prompts are implemented in `matformer/models/pyg_att.py`:
a node-level prompt (`NodePrompt`) is added to the node features and a graph-level
prompt embedding is injected, while the pretrained backbone stays frozen.

## ⚙️ Installation

```bash
pip install -r requirements.txt
```

> The pinned versions target **PyTorch 2.1.1 + CUDA 12.1** (with DGL and
> PyTorch Geometric wheels for the same build). Adjust the index URLs in
> `requirements.txt` if you use a different CUDA / PyTorch version.

## 🚀 Training

From inside the `matformer/` folder:

```bash
cd matformer

# JARVIS-DFT properties (formation energy, band gap, bulk/shear modulus, ...)
python train_jv.py

# Materials Project (MEGNet) properties (e_form, gap, bulk/shear modulus)
python train_mp.py
```

Select the target property by editing the `prop` variable in the corresponding
entry script, or call `train_prop_model(...)` from `train_props.py` directly to
customize the learning rate, batch size, number of epochs, and dataset.

