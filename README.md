# Image Classification Using Deep Learning Techniques on the CIFAR-10

This repository contains a Python application for classifying images using a pre-trained deep learning model. The application uses PyQt5 for the GUI and TensorFlow for model inference.

## Requirements

- Python 3.11.6
- PyQt5
- Pillow
- NumPy
- TensorFlow

**Note: This code is designed to run without errors only in Python 3.11.6 version.**

## Installation

1. Install the required packages:
    ```bash
    pip install PyQt5 Pillow numpy tensorflow
    ```
2. Download the pre-trained model [here](https://drive.google.com/drive/folders/1sWTD2RgQHXCxLnaZD2C6RQqrQDZKbfHC?usp=sharing) and place `cifar10_model.h5` in the project directory.

## Usage

1. Ensure that the pre-trained model `cifar10_model.h5` is in the project directory.

2. Run the application:
    ```bash
    python main.py
    ```

3. Use the GUI to select an image and classify it.

## Model Training (Required)

Training the model is optional. The application comes with a pre-trained model, but if you wish to train the model yourself, follow these steps:

1. Download and extract the dataset from [Kaggle](https://www.kaggle.com/datasets/arjuntejaswi/plant-village).

2. Update the paths in `model_training.py` to point to your dataset directories.

3. Run the training script:
    ```bash
    python Model_Training_Code.py
    ```

4. The trained model will be saved as `cifar10_model.h5`.
