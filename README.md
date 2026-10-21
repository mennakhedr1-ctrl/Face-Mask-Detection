# Face-Mask-Detection
CNN that classifies faces as with or without a mask, using OpenCV preprocessing and face detection. Built with TensorFlow/Keras.

## Overview
This project trains a CNN from scratch to classify face images as **with mask** or **without mask**. Images are resized to 128x128 and normalized. batch normalization, dropout, early stopping, and learning-rate reduction are used to control overfitting and stabilize training. The model is evaluated on a held-out test set using precision, recall, and F1-score, not accuracy alone.

## Approach
1. Loaded and resized all images to 128x128 RGB
2. Created labels and split the data with a stratified train/test split
3. Built a CNN with three Conv blocks (Conv2D, BatchNormalization, MaxPooling), followed by dense layers with dropout
4. Trained with early stopping (best weights restored) and ReduceLROnPlateau
5. Evaluated on the held-out test set
6. Added OpenCV Haar-cascade face detection for inference on new images, falling back to the whole image when no face is found

## Results (held-out test set, 756 images)
| Class | Precision | Recall | F1-score | Support |
|-------|-----------|--------|----------|---------|
| without_mask | 0.956 | 0.969 | 0.962 | 383 |
| with_mask | 0.967 | 0.954 | 0.961 | 373 |

| Metric | Score |
|--------|-------|
| Test accuracy | 96.2% |
| Macro avg F1 | 0.962 |

## Tech Stack
Python, TensorFlow, Keras, OpenCV, Scikit-learn, NumPy, Matplotlib, Streamlit
 
