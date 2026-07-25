# DPSIC: Dual-Path Self-Paced Incremental Clustering for Large-Scale High-Dimensional Data

This repository provides the implementation of **DPSIC**, a unified dual-path self-paced incremental clustering framework for large-scale high-dimensional data.

---
## Repository Structure
DPSIC/
│
├── README.md
│
├── code/
│ ├── demo_TIME.py
│ ├── Sample.py
│ ├── RL_TIME.py
│ └── base1.py
│
├── preprocessing/
│ └── preprocess_dataset.py
│
└── requirements.txt

---
## Requirements

The implementation is developed with Python.
Main dependencies:
- numpy
- scipy
- scikit-learn
- h5py
- pandas
- pytorch
- 
Install dependencies:
```bash
pip install -r requirements.txt

---
## Dataset Preparation
Due to the large size of these datasets, the complete datasets are not included in this repository.
The original datasets can be obtained from their corresponding public sources:
Dataset 1: [Dataset URL]
Dataset 2: [Dataset URL]
Dataset 3: [Dataset URL]

STRA dataset:
Run the preprocessing script:
python preprocessing/preprocess_dataset.py
The generated dataset should be stored as:
dataset_name.h5


