# Pneumonia Detection with Lightweight CNN Models

### 🧐 I. Overview
This repository contains a project focused on identifying the presence of pneumonia in chest X-ray images. Each image can be classified into one of three categories: **Bacterial Pneumonia**, **Viral Pneumonia**, or **Normal**.

This project leverages 3 models trained via transfer learning. These models are adapted from the following pre-trained lightweight convolutional neural network (CNN) architectures to perform multiclass image classification:
- MobileNet-V2
- ShuffleNet-V2
- SqueezeNet 1.1

All models are trained using identical batch sizes, epochs, & data preprocessing techniques to ensure a fair comparison of their performance.

----------------------

### 🗂️ II. Dataset
![image](https://github.com/m3mentomor1/Pneumonia_Detection_with_Lightweight-CNN-Models/assets/95956735/ac6adea5-0215-4ee9-b20b-d64a56e9237c)

#### Chest X-Ray Images
- The dataset is re-structured into three main directories: **train**, **val**, & **test**. Within each directory, there are subfolders representing different image categories, namely **Bacterial Pneumonia**, **Viral Pneumonia**, & **Normal**. Altogether, the dataset comprises 4,353 chest X-ray images in JPEG format, distributed across the three classes in each set.
- These chest X-ray images were chosen from retrospective cohorts of pediatric patients aged 1-5 years old at the Guangzhou Women and Children’s Medical Center, Guangzhou. The chest X-ray imaging was conducted as part of the routine clinical care for these patients.

**Source:** D. Kermany, K. Zhang, and M. Goldbaum, “Labeled Optical Coherence Tomography (OCT) and Chest X-Ray Images for Classification,” data.mendeley.com, vol. 2, Jun. 2018, doi: https://doi.org/10.17632/rscbjbr9sj.2.

**Download Dataset Here:** 
https://www.kaggle.com/datasets/paultimothymooney/chest-xray-pneumonia/

Re-structured | https://drive.google.com/drive/folders/17RAWWpF2voDNdMZxU-wXoiMBxCQO2b09?usp=sharing

----------------------

### 🚀 III. Model Deployment
<img src="https://github.com/m3mentomor1/Breast-Cancer-Image-Classification-with-DenseNet121/assets/95956735/6d0001fd-6890-44aa-8f25-223b21e8ab39" width="300" />

- a free and open-source Python framework to rapidly build and share beautiful machine learning and data science web apps. 
##
The model is deployed on **Streamlit**, allowing for a straightforward and accessible user interface where users can conveniently do pneumonia detection.

**Access the app here:** https://pneumoniadetectionwithlightweight-cnn-models-6sgynwffygezemyf8.streamlit.app/









