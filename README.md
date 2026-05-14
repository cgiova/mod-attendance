<p align="center">
  <a href="" rel="noopener">
 <img src="logo.png" alt="Project logo" width="300" height="350"></a>
</p>

<h3 align="center">ISIC 24</h3>

<div align="center">

[![Status](https://img.shields.io/badge/status-active-success.svg)]()
[![License](https://img.shields.io/badge/license-MIT-blue.svg)](/LICENSE)

</div>

---

<p align="center"> REFINING MULTIMODAL EXPLAINABILITY THROUGH MODALITY ATTENTIVE REGULARIZATION <br>

</p>

## 📝 Table of Contents

- [About](#about)
- [Getting Started](#getting_started)
- [TODO](TODO.md)
- [Authors](#authors)
- [Acknowledgments](#acknowledgements)

## 🧐 About <a name = "about"></a>
This project investigates how explicitly controlling modality attention in multimodal neural networks affects both model 
performance and the quality of generated explanations in a clinical decision support context. Using skin lesion images 
and tabular clinical metadata as input modalities, we introduce a framework of regularization strategies that allow 
deliberate, interpretable control over how visual and clinical features are weighted — both during training and at 
inference time. Rather than relying solely on the network's learned attention mechanism to balance modality contributions, 
our approach makes modality governance explicit and transparent. The goal is to evaluate whether this produces explanations 
that are more faithful, stable, and actionable for clinicians with no technical background, and to quantify any trade-offs 
with predictive performance.

## 🏁 Getting Started <a name = "getting_started"></a>

### Data

This repo contains code and notebooks. The ISIC 2024 dataset is expected to be available locally. 
Original dataset is available at the [isic 2024 official Kaggle challenge page](https://www.kaggle.com/competitions/isic-2024-challenge)

### Initializing
Set an environment variable pointing to the dataset root folder:

**macOS / Linux**
```bash
export ISIC24_DATA_DIR="/absolute/path/to/data"
```

### Installation <a name = "installation"></a>
This project is tested with **Python 3.12**.

#### macOS / Linux
```bash
python3.12 -m venv .venv
source .venv/bin/activate
```

#### Pip
From the repo root:
```bash
python -m pip install --upgrade pip setuptools wheel
python -m pip install -r requirements.txt
python -m pip install -e .
```

#### Conda/Mamba
From the repo root:
```bash
mamba env create -f environment.yml
mamba activate your_project
python -m pip install -e .
```


## 🎈 Usage <a name="usage"></a>

Add notes about how to use the system.

## ✍️ Authors <a name = "authors"></a>

- [@Claudio Giovannoni](https://github.com/cgiova)


## 🎉 Acknowledgements <a name = "acknowledgements"></a>

- Hat tip to anyone whose code was used
- Inspiration
- References
