# Vision Transformer for Lesion Detection in the Gastrointestinal Tract

## Abstract

Nowadays, at least one million patients with gastrointestinal problems are treated with VCE. Video Capsule Endoscopy is a powerful non-invasive diagnostic technique, which has recently become extremely popular, first developed to visualize the small intestine, it is now designed to examine all the anatomic sites of the gastrointestinal tract, such as oesophagus, stomach, and cecum. However, an unsought drawback of this technology is the production of long-lasting videos because they have to naturally cross the digestive tract. An experienced endoscopist, who analyses the collected frames, requires at least two hours of work using semi-automatic help tools.
Hence it turns out to be time-consuming and limited for large-scale analysis. The established progress of deep-learning methods to effectively support medical decisions suggests the need to follow this rising trend, by proposing a novel AI-based lesion classifier, which can distinguish whether a lesion is present or not in a given frame and to identify the type of lesion: e.g., ulcers, bleedings, and polyps. The classifier under discussion leverages the most recent success of Transformers, that have been shown to generalize across a wide variety of tasks, including vision ones. Vision Transformers have shown that it is possible to use a network originally created to perform natural language processing tasks, which operates on words, to learn from images instead. Therefore, it might turn the tide of the already settled and deep-rooted fame of the state-of-the-art.

## Table of Contents

- [Introduction](#introduction)
- [Getting Started](#getting-started)
  - [Prerequisites](#prerequisites)
- [Usage](#usage)
- [Training](#training)
- [Results](#results)

## Introduction

This project leverages Vision Transformers, a deep learning architecture used generally for Natural Language Processing, to classify images into 8 predefined categories of lesions. The code supports multi-class classification and provides instruction for both training and evaluation.

## Getting Started

### Prerequisites

Before you begin, ensure you have met the following requirements:

- Python 3.7 or higher
- PyTorch 1.6 or higher
- Other dependencies specified in `requirements.txt`

### Installation

To install the required dependencies, follow these steps:

1. Clone the repository:

   ```shell
   git clone https://github.com/marinoalessio/vit_for_lesion_detection_in_GI_tract.git
   ```

2. Install dependencies:

   ```shell
   pip install -r requirements.txt
   ```

## Usage

To use this code for image classification, follow these steps:

1. Prepare your dataset and organize it according to your requirements.

2. Configure the training parameters in the code, such as the dataset paths, model type, batch size, and other hyperparameters.

3. Train the model using the `train` function. You can specify options like the number of training steps, learning rate, and optimizer type.

4. Evaluate the model using the `valid` function on the validation and test datasets. You can choose the type of accuracy to compute (simple, balanced, or both).

## Training

Training the model involves running the `train` function with appropriate configurations. Here are the basic steps:

1. Load and preprocess your dataset.

2. Initialize the ViT model with the desired configuration.

3. Specify the training parameters, such as the number of epochs, learning rate, and batch size.

4. Start training the model.

For example, you can run training with the following command:

```shell
python train.py --name image_classification --dataset MRI --model_type ViT-B_16 --num_fold 1
```

## Results

The ultimate accuracy score is 75%, calculated on the test accuracy at which validation accuracy reaches its peak of 83%.


---
