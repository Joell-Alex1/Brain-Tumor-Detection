Brain Tumor Detection using Convolutional Neural Networks (CNN)

This project builds and evaluates a Convolutional Neural Network (CNN) to classify brain MRI scans as either tumor or non-tumor. The notebook includes the complete pipeline: dataset download, preprocessing, training, evaluation, and prediction on custom images.

Project Overview

This repository contains a Jupyter Notebook that demonstrates:

Downloading the MRI dataset using KaggleHub

Loading and preprocessing images

Building a CNN model using TensorFlow/Keras

Training the model with proper train–test splitting

Evaluating performance using accuracy, loss, classification report, and confusion matrix

Testing the model on both dataset images and user-uploaded MRI images

The project uses grayscale MRI images and performs binary classification (tumor vs. no tumor).

Features

Automated dataset download using KaggleHub

Image preprocessing with OpenCV

CNN architecture with multiple convolution and pooling layers

Training visualization (accuracy and loss plots)

Confusion matrix and classification report

Single-image prediction helper function

User-upload support for testing custom MRI scans

Technologies Used

Python

TensorFlow / Keras

OpenCV

NumPy

Scikit-learn

Matplotlib

Seaborn

KaggleHub

Dataset

Dataset used:
Brain MRI Images for Brain Tumor Detection
Downloaded automatically using:
kagglehub.dataset_download("navoneel/brain-mri-images-for-brain-tumor-detection")

Directory structure after download:

brain_tumor_dataset/
   yes/
   no/

Model Architecture

The CNN includes:

Three convolutional blocks (Conv2D + MaxPooling)

Flatten layer

Dense layer with ReLU activation

Dropout layer to prevent overfitting

Output layer with sigmoid activation

Loss function: Binary Crossentropy
Optimizer: Adam

Training

The notebook trains the model for 20 epochs with:

Batch size: 32

80/20 train–test split

Stratified sampling to preserve class distribution

Plots for accuracy and loss are generated.

Evaluation

Metrics used:

Accuracy (Train and Test)

Validation curves

Classification report

Confusion matrix

Single Image Prediction

A helper function predict_single_image() allows prediction on any given MRI image file.
The notebook also supports manual file upload (via Google Colab) for real-time testing.

How to Use

Open the notebook in Google Colab or Jupyter.

Run all cells sequentially.

Upload an MRI image at the end to see the model’s prediction.

File Structure
Brain_Tumor_Detection.ipynb   # Main project notebook
README.md                     # Project documentation

Notes

This project is intended for educational and experimental purposes.

The model is not meant for clinical use or real medical diagnosis.
