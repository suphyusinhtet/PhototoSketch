# Photo to Sketch Conversion using Autoencoder 

This project focuses on converting photos into sketch-like images using an **autoencoder model**. The autoencoder is trained on a dataset of paired photo and sketch images. This project demonstrates how neural networks can be used to learn an unsupervised mapping from photos to their sketch equivalents.

#### This model has some limitations. It works better for portrait face images with clear background, Try colab!!

#### **Small-Model(256 size)**
[<img src="https://colab.research.google.com/assets/colab-badge.svg" align="center">](https://colab.research.google.com/github/suphyusinhtet/PhototoSketch/blob/main/256sizeModel.ipynb)

#### **Big-Model(512 size)**
[<img src="https://colab.research.google.com/assets/colab-badge.svg" align="center">](https://colab.research.google.com/github/suphyusinhtet/PhototoSketch/blob/main/512sizeModel.ipynb)

> I uploaded the trained models for both 256 and 512 sizes of output images.

### Link for testing - [Click here](https://phototosketch-xj3dfvjkvye49huorlqrr7.streamlit.app/)

## Table of Contents
1. [Overview](#overview)
2. [How Autoencoder Model Works](#how-autoencoder-model-works)
3. [Project Pipeline](#project-pipeline)
4. [Results](#results)
5. [Conclusion](#conclusion)
6. [Limitations](#limitations)
7. [Future Work](#future-work)
8. [Acknowledgements](#acknowledgements)

## Overview

The primary goal of this project is to design an autoencoder that can take an input photo and generate a corresponding sketch. The project uses data augmentation techniques to expand the dataset and train the model more effectively. The autoencoder learns to compress the images into a lower-dimensional space and reconstruct them as sketches. 

### Dataset
![Dataset](Dataset-1.JPG)
- **Photos**: RGB images of real-world scenes.
- **Sketches**: Hand-drawn sketch-like images corresponding to the photos.
- **Data Augmentation**: Techniques like flipping, rotating, and resizing are applied to increase the size of the dataset.

The dataset is divided into:
- 320 images for training.
- 80 images for testing.

## How Autoencoder Model Works

An autoencoder is a type of neural network used to learn efficient codings of unlabeled data. It consists of two parts:
1. **Encoder**: Compresses the input into a lower-dimensional latent space.
2. **Decoder**: Reconstructs the input from the compressed latent representation.

### Encoder
The encoder part of the network consists of **convolutional layers** which downsample the image by reducing its spatial dimensions while capturing important features. It performs:
- **Convolution**: Extracts features from the input images.
- **Leaky ReLU Activation**: Provides non-linearity, allowing the model to learn complex representations.
- **Batch Normalization**: Normalizes the activations, stabilizing training.

### Decoder
The decoder part of the autoencoder upsamples the encoded image, reconstructing it into the desired output. It consists of **transpose convolution layers** to increase the image dimensions back to the original size. The decoder is responsible for generating the sketch version of the input photo.

### Model Training
The model is compiled using the **Adam optimizer** and trained using **mean absolute error (MAE)** as the loss function. Early stopping is used to halt training when no improvement in the loss is observed.

```python
model.compile(optimizer = tf.keras.optimizers.Adam(learning_rate = 0.001), loss = 'mean_absolute_error', metrics = ['acc'])
```
## Project Pipeline

### Data Preprocessing:
- Load photo and sketch pairs.
- Resize images to `256x256` and `512x512`
- Normalize pixel values to the range `[0,1]`.
- Apply data augmentation: flipping, rotation.

### Model Architecture:
- **Encoder**: Three convolutional layers for downsampling the image.
- **Decoder**: Three transpose convolutional layers to reconstruct the image.
- **Activation functions**: Leaky ReLU and Sigmoid.

### Training:
- Train the model using the training set.
- Use early stopping to prevent overfitting.

### Evaluation:
- Evaluate the model on the test set.
- Display the original photo, actual sketch, and predicted sketch side-by-side.

## Results
The model successfully learned to convert photos into sketch images. The predicted sketches are visually similar to the actual sketches from the dataset.

### Model Performance:
- **Loss**: The final loss value was observed as `0.085` (Mean Absolute Error).
- **Accuracy**: The final accuracy value was `62.8%`.

Example of the results:


## Conclusion
In this project, an autoencoder model was successfully implemented to convert photos into sketch images. The model was able to capture essential details from the input photos and translate them into sketch-like images. This demonstrates the capability of convolutional neural networks in learning the mapping between two different visual domains.

### Key Takeaways:
- The autoencoder model is effective for tasks involving image-to-image translation.
- Data augmentation techniques are crucial to improving the model’s generalization ability.

## Limitations
- **Image Quality**: The model may struggle to generate high-quality sketches when the input photo has complex textures or features.
- **Training Time**: Training the model on a large dataset can be time-consuming, especially with limited hardware resources.
- **Generalization**: The model is trained on a specific dataset and may not generalize well to different photo styles or lighting conditions.

## Future Work
Several areas for improvement can be explored in future iterations of the project:
- **Advanced Architectures**: Exploring more complex architectures such as U-Net or Generative Adversarial Networks (GANs) to improve the quality of generated sketches.
- **Larger Datasets**: Training on larger and more diverse datasets to improve the generalization capability of the model.
- **Real-Time Sketch Generation**: Developing a real-time application that can generate sketches instantly from a live camera feed.
- **Improved Loss Functions**: Experimenting with perceptual loss or content loss to make the generated sketches more visually appealing.

## Acknowledgements
I would like to thank the open-source community for providing datasets and tools that made this project possible. Special thanks to:
- **TensorFlow** and **Keras** for providing the deep learning framework.
- **Kaggle** for hosting various image datasets.
- **OpenCV** for image processing tools.


