# Marine Animal Image Classification

This repository contains code and resources for image classification of marine animals using Convolutional Neural Networks (CNNs). The project includes training multiple CNN architectures and fine-tuning a pre-trained model (VGG16) for classifying images of dolphins, jellyfish, sea rays, starfish, and whales.

## Features

- Implementation of two custom CNN architectures.
- Fine-tuning of the VGG16 pre-trained model.
- Comparison of model performance on test data.
- Evaluation of predictions on custom images.

## Table of Contents

1. [Project Overview](#project-overview)
2. [Dataset](#dataset)
3. [Model Architectures](#model-architectures)
4. [Training and Evaluation](#training-and-evaluation)
5. [Results](#results)
---

## Project Overview

This project explores CNN-based image classification to identify marine animals. The objective was to build models with high accuracy while minimizing overfitting. Transfer learning was leveraged to improve performance.

## Dataset

The dataset consists of images of the following classes:
- Dolphin
- Jellyfish
- Sea Rays
- Starfish
- Whale

The images were scaled and normalized to prepare them for model training.

## Model Architectures

### Custom CNN Models
1. **My_CNN**: Three convolutional layers, max-pooling, dropout, and a softmax classifier.
2. **My_CNN2**: Similar to My_CNN but with more filters and a slightly modified architecture.

### Fine-Tuned Model
- **My_Fine_Tuned**: Based on the pre-trained VGG16 model with modifications for the marine animal classification task.

## Training and Evaluation

- **Training**: The custom models were trained for 25 epochs each, and the fine-tuned model for 5 epochs.
- **Metrics**: Accuracy and categorical cross-entropy were used for evaluation.
- **Results**:
  - My_CNN: 59.5% test accuracy.
  - My_CNN2: 56.8% test accuracy.
  - My_Fine_Tuned: 77.6% test accuracy.

## Results

The models' performance is summarized below:

| Model          | Test Accuracy |
|----------------|---------------|
| My_CNN         | 59.5%         |
| My_CNN2        | 56.8%         |
| My_Fine_Tuned  | 77.6%         |
