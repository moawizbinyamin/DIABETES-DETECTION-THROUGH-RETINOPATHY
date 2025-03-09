# INFYMA Hybrid CNN+ViT

## Overview
This repository contains a Jupyter Notebook implementing a hybrid deep learning model that combines Convolutional Neural Networks (CNN) and Vision Transformers (ViT) for diabetic retinopathy classification. The approach leverages feature extraction from both architectures to improve classification performance.

## Features
- Utilizes ResNet for CNN-based feature extraction.
- Incorporates a Vision Transformer (ViT) to capture global attention-based features.
- Performs feature concatenation for a hybrid representation.
- Applies feature selection techniques to refine the final input to the classifier.
- Trains a classification model for diabetic retinopathy detection.
- Includes visualization of feature importance.

## Dependencies
The notebook requires the following Python libraries:

```bash
pip install torch torchvision timm numpy pandas scikit-learn matplotlib seaborn
```

## Files
- **INFYMA Hybrid CNN+ViT.ipynb**: Main notebook implementing the hybrid model.
- **data/**: Directory where diabetic retinopathy images should be stored.
- **models/**: Directory to save trained models.

## Usage
1. Clone the repository:
   ```bash
   git clone https://github.com/moawizbinyamin/DIABETES-DETECTION-THROUGH-RETINOPATHY
   cd your-repo
   ```
2. Open the Jupyter Notebook:
   ```bash
   jupyter notebook "INFYMA Hybrid CNN+ViT.ipynb"
   ```
3. Run the notebook cell by cell to train and evaluate the model.

## Results
The notebook includes evaluation metrics such as accuracy, precision, recall, and confusion matrices. Visualizations of extracted features and their importance are also provided.

## Acknowledgments
- **ResNet**: Pre-trained model used for feature extraction.
- **Vision Transformer (ViT)**: Transformer-based model for image processing.
- Open-source datasets for diabetic retinopathy classification.



