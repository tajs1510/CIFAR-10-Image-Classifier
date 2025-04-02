# Image Recognition Using Deep Learning Models on CIFAR-10 Dataset

## Project Overview
This project implements and compares two deep learning models (CNN and ResNet) for image classification using the CIFAR-10 dataset. The work was completed as part of the "Introduction to Data Analysis and Deep Learning" course at Van Lang University.

## Dataset
CIFAR-10 dataset contains 60,000 32x32 color images in 10 classes:
- 50,000 training images
- 10,000 test images

Classes: airplane, automobile, bird, cat, deer, dog, frog, horse, ship, truck

Dataset source: [CIFAR-10 Official Website](http://www.cs.toronto.edu/~kriz/cifar.html)

## Models Implemented

### 1. CNN (Convolutional Neural Network)
- Architecture:
  - 3 Convolutional layers (32, 64, 128 filters)
  - Batch Normalization
  - Max Pooling
  - Fully Connected layer (128 units)
  - Dropout (0.5)
  - Output layer (10 units with softmax)

- Results:
  - Training accuracy: ~85%
  - Validation accuracy: ~85%

### 2. ResNet (Residual Network)
- Architecture:
  - Residual blocks with skip connections
  - 64, 128, 256 filters in successive blocks
  - Global Average Pooling
  - Output layer (10 units with softmax)

- Results:
  - Training accuracy: ~90%
  - Validation accuracy: ~90%


