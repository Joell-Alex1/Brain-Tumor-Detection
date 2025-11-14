Brain Tumor Detection using Convolutional Neural Networks (CNN)

This project implements a Convolutional Neural Network (CNN) to classify brain MRI scans as either tumor or non-tumor. It includes dataset download, preprocessing, model training, evaluation, and single-image prediction.

Project Overview

This repository contains a Jupyter Notebook demonstrating:

Downloading the MRI dataset using KaggleHub

Loading and preprocessing MRI images

Building a CNN using TensorFlow/Keras

Training with a proper train–test split

Evaluating with accuracy, loss curves, classification report, and confusion matrix

Testing predictions on dataset images and custom-uploaded MRI scans

The project performs binary classification on grayscale MRI images.

Features

Automatic dataset download

Image preprocessing using OpenCV

CNN with multiple convolution + pooling layers

Training visualizations

Classification report & confusion matrix

Single-image prediction function

Support for uploading custom MRI images

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

Dataset: Brain MRI Images for Brain Tumor Detection
Downloaded using:

kagglehub.dataset_download("navoneel/brain-mri-images-for-brain-tumor-detection")


Directory structure:

brain_tumor_dataset/
   yes/
   no/

Model Architecture

The CNN includes:

3 × Conv2D + MaxPooling layers

Flatten layer

Dense layer (ReLU)

Dropout for regularization

Output layer with Sigmoid activation

Loss: Binary Crossentropy
Optimizer: Adam

Training

20 epochs

Batch size: 32

80/20 train–test split

Stratified sampling to preserve class distribution

Training accuracy and loss curves are generated.

Evaluation

The notebook evaluates the model using:

Accuracy

Loss curves

Classification report

Confusion matrix

Single Image Prediction

A helper function allows prediction on any MRI image:

predict_single_image("path_to_image.jpg")


You can also upload images directly in the notebook (Google Colab).

How to Use

Open the notebook in Jupyter or Google Colab

Run all cells sequentially

Upload an MRI image to test the model's prediction

File Structure
Brain_Tumor_Detection.ipynb   # Main notebook
README.md                     # Project documentation

Notes

This project is intended for educational and experimental purposes only.
It is not suitable for real medical diagnosis.
