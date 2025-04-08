# Alzheimer’s Disease Detection using CNN on MRI Scans

[![License: MIT](https://img.shields.io/badge/License-MIT-blue.svg)](LICENSE)
[![Python Version](https://img.shields.io/badge/Python-3.8%2B-blue.svg)](https://www.python.org/)

A deep learning project for the classification of Alzheimer's disease from MRI scans of the left hippocampus—enhanced with advanced signal processing techniques for superior medical image analysis.

---

## Table of Contents

- [Overview](#overview)
- [Project Structure](#project-structure)
- [Signal Processing Techniques](#signal-processing-techniques)
- [Data Description](#data-description)
- [Technical Implementation](#technical-implementation)
- [Results](#results)
- [Future Improvements](#future-improvements)
- [Notes](#notes)

---

## Overview

This project leverages a **Convolutional Neural Network (CNN)** optimized for 3D medical imaging to detect Alzheimer’s disease. By integrating advanced signal processing methods, the model enhances the quality and interpretability of MRI scans, providing a robust tool for early diagnosis.

---

## Project Structure

The repository is divided into three main components:

1. **Patient Data Processing**  
   *File: [`Part_I_Patient_data.ipynb`](Part_I_Patient_data.ipynb)*  
   - Preprocessing and feature engineering for demographic and clinical data  
   - Integration with MRI image data  
   - Signal quality assessment and validation  

2. **MRI Data Processing**  
   *File: [`Part_II_MRI_data.ipynb`](Part_II_MRI_data.ipynb)*  
   - Preprocessing of MRI scans  
   - Signal processing pipeline:
     - **Noise Reduction:** Gaussian filtering, Median filtering, Adaptive thresholding  
     - **Image Normalization:** Standardization, Contrast enhancement  
     - **Feature Extraction:** Edge detection, Texture analysis, Wavelet transforms  
   - Data augmentation: rotations, intensity variations, spatial transformations

3. **CNN Implementation**  
   *File: [`Part_III_CNN.ipynb`](Part_III_CNN.ipynb)*  
   - Building and training the CNN model architecture  
   - Model evaluation and testing  
   - Integration of signal processing into the model pipeline

---

## Signal Processing Techniques

### Preprocessing Pipeline
- **Noise Reduction:**  
  - Gaussian filtering to suppress noise  
  - Median filtering to remove salt-and-pepper artifacts  
  - Adaptive thresholding for dynamic enhancement

- **Image Enhancement:**  
  - Histogram equalization for contrast improvement  
  - Intensity normalization (z-score standardization)  
  - Spatial normalization to a standard anatomical template

- **Feature Extraction:**  
  - Edge detection using Sobel and Canny operators  
  - Texture analysis via Gabor filters  
  - Multi-scale analysis with wavelet transforms  
  - Texture classification with local binary patterns

- **Data Augmentation:**  
  - Random rotations (±15°)  
  - Intensity scaling (±20%)  
  - Elastic deformations  
  - Random cropping and padding

---

## Data Description

- **Input:** 3D MRI scans of the left hippocampus  
- **Dimensions:** `30 x 40 x 30` tensors  
- **Signal Processing Parameters:**
  - Sampling rate: *[Specify if available]*
  - Resolution: *[Specify if available]*
  - Optimized for signal-to-noise ratio enhancement  
- **Classification Classes:**
  - **Cognitively Normal (False)**
  - **Alzheimer's Disease (True)**

---

## Technical Implementation

### Dependencies
- **Deep Learning:** PyTorch
- **Cloud Environment:** Google Colab
- **Image & Signal Processing Libraries:**
  - SciPy  
  - OpenCV  
  - scikit-image  
- **Data Science Tools:** NumPy, Pandas, etc.

### Model Architecture
- **Input Tensor Size:** 30 x 40 x 30  
- **Output:** Binary classification (Alzheimer's vs. Cognitively Normal)  
- **Key Features:**
  - Custom convolutional layers with specialized kernels  
  - Pooling layers to reduce dimensionality  
  - Batch normalization to stabilize the signal  
  - Integrated signal processing layers for improved feature extraction

### Data Pipeline
1. **Data Loading:** Retrieve MRI data from Google Drive  
2. **Preprocessing:** Apply signal enhancement and noise reduction  
3. **Feature Extraction:** Isolate significant image features  
4. **Data Splitting:** Create training and validation sets  
5. **Model Training:** Optimize training with continuous validation feedback  
6. **Model Testing:** Evaluate using a comprehensive testing framework

---

## Results

The CNN model achieves robust binary classification accuracy in detecting Alzheimer's disease by effectively leveraging advanced signal processing techniques integrated within the training pipeline.

---

## Future Improvements

- **Multi-Modality Integration:** Incorporate additional medical imaging data (e.g., PET, CT)  
- **Enhanced Signal Processing:**  
  - Deep learning–based denoising methods  
  - Advanced wavelet transforms  
  - Non-local means filtering and super-resolution techniques  
- **Model Ensemble:** Develop ensemble methods to boost accuracy  
- **Explainability:** Incorporate explainable AI frameworks for better clinical interpretability  
- **Real-Time Optimization:** Optimize the processing pipeline for real-time data analysis

---

## Notes

- **Dataset:** MRI scans are stored in Google Drive; ensure access permissions are correctly configured.  
- **Data Format:** Training data provided in `.pt` format (PyTorch tensors).  
- **Customization:** Signal processing parameters are configurable to cater to different imaging conditions and research needs.

---

Happy coding and best of luck with advancing Alzheimer’s diagnostics!

