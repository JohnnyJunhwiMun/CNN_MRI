# Alzheimer’s Disease Detection using CNN on MRI Scans


A deep learning project for the binary classification of Alzheimer's disease from MRI scans of the left hippocampus—enhanced with advanced signal processing techniques for superior medical image analysis.

---

## Table of Contents

- [Overview](#overview)
- [Project Structure](#project-structure)
- [Signal Processing Techniques](#signal-processing-techniques)
- [Data Description](#data-description)
- [Technical Implementation](#technical-implementation)
- [Results](#results)


---

## Overview

This project leverages a **Convolutional Neural Network (CNN)** designed for 3D medical imaging to detect Alzheimer’s disease. By incorporating sophisticated signal processing methods—including image preprocessing, spatial normalization, and feature extraction—the model improves the quality and interpretability of MRI scans, ultimately supporting enhanced diagnostic accuracy.

---

## Project Structure

The repository is organized into three main components:

1. **Patient Data Processing**  
   *File: [`Part_I_Patient_data.ipynb`](Part_I_Patient_data.ipynb)*  
   - Processes patient demographic and clinical data  
   - Performs data cleaning, feature engineering, and integration with MRI data

2. **MRI Data Processing**  
   *File: [`Part_II_MRI_data.ipynb`](Part_II_MRI_data.ipynb)*  
   - Loads and preprocesses MRI scans in NIfTI-1 format (`.nii.gz`)  
   - Utilizes `scipy.ndimage` for signal processing  
   - Visualizes data, extracts features, and applies affine spatial transformations

3. **CNN Implementation**  
   *File: [`Part_III_CNN.ipynb`](Part_III_CNN.ipynb)*  
   - Defines the CNN model architecture optimized for 3D imaging  
   - Manages training, validation, and testing  
   - Integrates signal processing techniques directly into the model pipeline

---

## Signal Processing Techniques

### Preprocessing Pipeline

- **Data Loading & Header Analysis:**  
  Load MRI scans in NIfTI-1 format and analyze header information.

- **Image Preprocessing:**  
  - Extract image data and apply noise reduction techniques using `scipy.ndimage`
  - Perform affine spatial transformation to standardize image orientations

- **Feature Extraction & Normalization:**  
  - Extract significant imaging features for diagnostic relevance  
  - Normalize features to optimize the input for CNN processing

---

## Data Description

- **Input:** 3D MRI scans of the left hippocampus  
- **Format:** NIfTI-1 (`.nii.gz`)  
- **Dimensions:** 256 x 256 x 160 voxels  
- **Voxel Size:** 1 x 1 x 1 mm  
- **Data Type:** int16  
- **Classification Labels:**
  - **Cognitively Normal (False)**
  - **Alzheimer's Disease (True)**

---

## Technical Implementation

### Dependencies

- **Deep Learning:** PyTorch  
- **Cloud Environment:** Google Colab  
- **Signal Processing & Neuroimaging Libraries:**
  - `scipy.ndimage` for image processing  
  - `nibabel` for handling NIfTI-1 files  
  - `nilearn` for neuroimaging data analysis  
- **Data Science Tools:** NumPy, Pandas, etc.

### Model Architecture

The CNN is tailored for 3D medical imaging with the following features:
- **Input Tensor Size:** 30 x 40 x 30  
- **Output:** Binary classification (Alzheimer's vs. Cognitively Normal)  
- **Key Components:**
  - Custom convolutional layers with specialized kernels  
  - Pooling layers for dimensionality reduction  
  - Batch normalization to stabilize the training process  
  - Integrated signal processing layers for enhanced feature extraction


---

## Results

The CNN model demonstrates robust binary classification performance for Alzheimer's disease detection. With the integration of advanced signal processing techniques, the model significantly improves the accuracy and reliability of interpreting 3D MRI scans of the left hippocampus.

---

Happy coding and best of luck in advancing Alzheimer’s diagnostics!
