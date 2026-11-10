# CNN-Based Image Classification Using Deep Learning

## 1. Project Overview

This project implements a Convolutional Neural Network (CNN) from scratch to classify natural scene images into six different categories using the Intel Image Classification Dataset.

The model identifies the following scene categories:

- Buildings
- Forest
- Glacier
- Mountain
- Sea
- Street

## 2. Objectives

- Explore and understand the image dataset.
- Perform image preprocessing and normalization.
- Apply data augmentation to improve model generalization.
- Build a CNN architecture with at least three convolutional layers.
- Train and validate the CNN model.
- Evaluate model performance using classification metrics.
- Visualize predictions and model performance.

## 3. Dataset

**Dataset:** Intel Image Classification Dataset

**Source:** https://www.kaggle.com/datasets/puneet6060/intel-image-classification

The dataset contains six image categories.

Dataset folders:

- `seg_train`: Training and validation images.
- `seg_test`: Testing images.
- `seg_pred`: Unlabelled images for additional prediction.

## 4. Technologies Used

- Python
- TensorFlow / Keras
- NumPy
- Pandas
- Matplotlib
- Seaborn
- Scikit-learn
- Pillow

## 5. Data Preprocessing

The following preprocessing operations were performed:

- Resized images to 150 × 150 pixels.
- Used RGB image format.
- Normalized pixel values to the range 0–1.
- Split training data into training and validation subsets.
- Used batches of 32 images.

## 6. Data Augmentation

The following augmentation techniques were applied:

- Random horizontal flipping.
- Random rotation.
- Random zoom.
- Random translation.

These techniques help the model learn from variations in images and reduce overfitting.

## 7. CNN Architecture

The CNN was built from scratch using TensorFlow/Keras.

The architecture includes:

1. Input layer.
2. Image rescaling layer.
3. Data augmentation layers.
4. Three convolutional layers with ReLU activation.
5. Max pooling layers.
6. Global average pooling layer.
7. Fully connected dense layer.
8. Dropout layer.
9. Softmax output layer with six classes.

The Adam optimizer and sparse categorical cross-entropy loss function were used.

## 8. Model Training

The model was trained using training data and monitored using validation data.

The following techniques were used:

- Early stopping.
- Model checkpointing.
- Training and validation accuracy monitoring.
- Training and validation loss monitoring.

## 9. Model Evaluation

The trained model was evaluated using the separate test dataset.

Evaluation metrics:

- Accuracy
- Precision
- Recall
- F1-score
- Confusion matrix

## 10. Visualizations

The project generates:

- Training and validation accuracy graph.
- Training and validation loss graph.
- Confusion matrix.
- Ten random test image predictions.

## 11. Project Structure

```text
cnn_project/
│
├── dataset/
├── src/
│   ├── data_exploration.py
│   ├── view_images.py
│   ├── data_loader.py
│   ├── model.py
│   ├── train.py
│   ├── evaluate.py
│   └── predict.py
│
├── models/
├── outputs/
├── requirements.txt
└── README.md
```

## 12. How to Run the Project

Install dependencies:

```bash
pip install -r requirements.txt
```

Explore the dataset:

```bash
python src/data_exploration.py
```

Train the model:

```bash
python src/train.py
```

Evaluate the model:

```bash
python src/evaluate.py
```

Visualize predictions:

```bash
python src/predict.py
```

## 13. Conclusion

This project demonstrates how Convolutional Neural Networks can be used for multi-class image classification. It covers the complete deep learning workflow, from dataset exploration and preprocessing to model development, training, evaluation, and prediction visualization.
