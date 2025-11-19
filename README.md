# dl_model_classification_finetune

Milestone 1, ViT applied

## Plant Classification with ResNet-50 and ViT-Base-Patch16-224

This project compares the performance of a ResNet-50 and a ViT-Base-Patch16-224 model on the Plants Classification dataset. 
The goal is to evaluate training from scratch, fine-tuning, and the impact of augmentation strategies.

## Dataset

We use the Plants Classification dataset from Kaggle, consisting of 30 classes and 30,000 images.
Two augmentation strategies (Loss and Blur) are applied once before training to create a fixed augmented dataset.
The dataset can be found here: [Dataset](https://www.kaggle.com/datasets/marquis03/plants-classification)
