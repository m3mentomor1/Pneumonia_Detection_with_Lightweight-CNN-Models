<div align="center">
  <h1>Pneumonia Detection with Lightweight CNN Models</h1>
</div>

### 🧐 I. Overview
This project compares the performance of three lightweight CNN models in identifying the presence of pneumonia in chest X-ray images, categorizing them into three classes: **Bacterial Pneumonia**, **Viral Pneumonia**, or **Normal**.

Each model employs a corresponding lightweight convolutional neural network (CNN) architecture fine-tuned through transfer learning to do multiclass image classification on chest X-ray images:
- MobileNet-V2
- ShuffleNet-V2
- SqueezeNet

For a fair evaluation, all models have been trained using identical batch sizes, epochs, and data preprocessing techniques. Additionally, while the architectures of MobileNetV2, ShuffleNetV2, and SqueezeNet vary in design principles and specific layers, the process of adapting them for chest X-ray image classification remains consistent across all three models. Specifically, only the final fully connected layer of each model is replaced to align with the number of classes in the dataset. This modification enables a fair comparison of the models' performances without further alteration to their respective architectures.

##

### 🗂️ II. Dataset
![image](https://github.com/m3mentomor1/Pneumonia_Detection_with_Lightweight-CNN-Models/assets/95956735/ac6adea5-0215-4ee9-b20b-d64a56e9237c)

#### Chest X-Ray Images
- The dataset is re-structured into three main directories: **train**, **val**, & **test**. Within each directory, there are subfolders representing different image categories, namely **Bacterial Pneumonia**, **Viral Pneumonia**, & **Normal**. Altogether, the dataset comprises 4,353 chest X-ray images in JPEG format, distributed across the three classes in each set.
- These chest X-ray images were chosen from retrospective cohorts of pediatric patients aged 1-5 years old at the Guangzhou Women and Children’s Medical Center, Guangzhou. The chest X-ray imaging was conducted as part of the routine clinical care for these patients.

**Source:** D. Kermany, K. Zhang, and M. Goldbaum, “Labeled Optical Coherence Tomography (OCT) and Chest X-Ray Images for Classification,” data.mendeley.com, vol. 2, Jun. 2018, doi: https://doi.org/10.17632/rscbjbr9sj.2.

**Download Dataset Here:** 
- [Kaggle (Unstructured)](https://www.kaggle.com/datasets/paultimothymooney/chest-xray-pneumonia/)
- [Re-structured](https://drive.google.com/drive/folders/17RAWWpF2voDNdMZxU-wXoiMBxCQO2b09?usp=sharing)

##

### 🛠️ IV. Use this repository

**1. Clone this repository.**

   Run this command in your terminal: 
   ```
   git clone https://github.com/m3mentomor1/Pneumonia_Detection_with_Lightweight-CNN-Models.git
   ```
(Optional: You can also ```Fork``` this repository.)

**2. Go to the repository's main directory.**

   Run this command in your terminal: 
   ```
   cd Pneumonia_Detection_with_Lightweight-CNN-Models
   ```

##

### 🚀 III. Model Deployment

The three models are deployed on a single web app on [**Streamlit**](https://streamlit.io/).

**Access the app here:** https://pneumoniadetectionwithlightweight-cnn-models-6sgynwffygezemyf8.streamlit.app/









