# dl_model_classification_finetune | Image Classification

This repository contains a series of experiments focused on fine-tuning Vision Transformer and CNN models for multi-class image classification. The work is organized into milestones, each exploring different architectural choices, training strategies, and data augmentation pipelines. The primary goal is to evaluate controlled modifications or comparisons to pretrained Vision Transformers and CNNs under consistent and reproducible training conditions.

---

## Repository Structure

```text
dl_model_classification_finetune
├─ milestone_1
│  ├─ augment.py
│  ├─ plot_info.py
│  ├─ train_amd.py
│  └─ train_nvidia.py
│
├─ milestone_2
│  ├─ augment_v2.py
│  ├─ custom_attention.py
│  ├─ matrix_attention.py
│  ├─ matrix_head.py
│  ├─ train_head_amd.py
│  └─ train_sat_full_nvidia.py
│
├─ milestone_3
│  └─ TBD
│
├─ .gitignore
├─ LICENSE
└─ README.md
```

## Plant Classification Dataset

We use the Plants Classification dataset from Kaggle, consisting of 30 classes and 30,000 images.
The dataset can be found here: [Dataset](https://www.kaggle.com/datasets/marquis03/plants-classification)
