<div align="center">
  <h1>Pneumonia Detection with Lightweight CNN Models</h1>
</div>

### 🧐 I. Overview
This project evaluates the effectiveness of three lightweight CNN models in detecting pneumonia in chest X-ray images, considering both performance & computational efficiency. The images are categorized into three classes: **Bacterial Pneumonia**, **Viral Pneumonia**, or **Normal**.

Each model employs a corresponding lightweight convolutional neural network (CNN) architecture fine-tuned through transfer learning to do multiclass image classification on a chest X-ray dataset:
- MobileNet-V2
- ShuffleNet-V2
- SqueezeNet

For a fair evaluation, all models have been trained using identical batch sizes, epochs, and data preprocessing techniques. Additionally, while the architectures of MobileNetV2, ShuffleNetV2, and SqueezeNet vary in design principles and specific layers, the process of adapting them for chest X-ray image classification remains consistent across all three models. Specifically, only the final fully connected layer of each model is replaced to align with the number of classes in the dataset. This modification enables a fair comparison of the models' performances without further alteration to their respective architectures.
<br><br>
##

### 🗂️ II. Dataset
![image](https://github.com/m3mentomor1/Pneumonia_Detection_with_Lightweight-CNN-Models/assets/95956735/ac6adea5-0215-4ee9-b20b-d64a56e9237c)

#### Chest X-Ray Images
The dataset is structured into three main directories: **train**, **val**, & **test**. Within each directory, there are subfolders representing different image categories, namely **Bacterial Pneumonia**, **Viral Pneumonia**, & **Normal**. Altogether, the dataset comprises 4,353 chest X-ray images in JPEG format, distributed across the three classes in each set:
- Training Set = 3559
  - Bacterial Pneumonia = 1230
  - Viral Pneumonia = 988
  - Normal = 1341
- Validation Set = 310
  - Bacterial Pneumonia = 108
  - Viral Pneumonia = 104
  - Normal = 98
- Test Set = 484
  - Bacterial Pneumonia = 172
  - Viral Pneumonia = 148
  - Normal = 164

These chest X-ray images were chosen from retrospective cohorts of pediatric patients aged 1-5 years old at the Guangzhou Women and Children’s Medical Center, Guangzhou. The chest X-ray imaging was conducted as part of the routine clinical care for these patients.

**Source:** D. Kermany, K. Zhang, and M. Goldbaum, “Labeled Optical Coherence Tomography (OCT) and Chest X-Ray Images for Classification,” data.mendeley.com, vol. 2, Jun. 2018, doi: https://doi.org/10.17632/rscbjbr9sj.2.

**Download Dataset Here:** 
- [Original](https://www.kaggle.com/datasets/paultimothymooney/chest-xray-pneumonia/)
- [Re-structured](https://drive.google.com/drive/folders/17RAWWpF2voDNdMZxU-wXoiMBxCQO2b09?usp=sharing)
<br><br>
##

### 💻 III. Tech Stack
``Python`` ``PyTorch`` ``scikit-learn`` ``Pandas`` ``NumPy`` ``Matplotlib`` ``Seaborn`` ``Azure ML Studio``
<br><br>
##

### 🧾 IV. Model Evaluation

#### **➜ Test Accuracy**
| Model         | Accuracy (%) |
|---------------|--------------|
| MobileNetV2   | 90.08        |
| ShuffleNetV2  | 88.43        |
| SqueezeNet    | 56.61        |

MobileNetV2 demonstrated the highest accuracy, indicating its strong ability to generalize & perform well on new data beyond the training & validation sets. In contrast, SqueezeNet exhibited the lowest accuracy, indicating challenges in effectively generalizing to new instances.
<br><br><br>
#### **➜ Training & Validation Accuracy per Epoch**
![image](https://github.com/m3mentomor1/Pneumonia_Detection_with_Lightweight-CNN-Models/assets/95956735/5d6d25b8-5b17-4993-bc1e-1fc5bcb9e11c)

The training accuracy of **MobileNetV2** steadily increases throughout the epochs, with validation accuracy showing fluctuations, reaching its highest point at later epochs, particularly around epochs 12 to 15. However, validation accuracy also fluctuates during the training process, notably dipping around epochs 4 to 6.

Similarly, **ShuffleNetV2** exhibits a gradual increase in training accuracy over the epochs, with validation accuracy fluctuating but generally maintaining a higher level compared to MobileNetV2. The fluctuations in validation accuracy are observed throughout the training epochs, particularly notable dips around epochs 4 and 14.

In contrast, **SqueezeNet** starts with lower training accuracy compared to MobileNetV2 and ShuffleNetV2, but it shows improvement over time. Validation accuracy remains relatively low throughout training, gradually increasing towards the later epochs, with notable improvements around epochs 6 to 9.

Overall, MobileNetV2 and ShuffleNetV2 consistently achieve higher validation accuracies than SqueezeNet, suggesting stronger generalization abilities. While SqueezeNet improves in training accuracy, it struggles to generalize to unseen data, reflected in its lower validation accuracy. This indicates that MobileNetV2 and ShuffleNetV2 are more robust models for the task at hand.
<br><br><br>
#### **➜ Confusion Matrix**

The values along the diagonal of each matrix represent the number of correctly predicted images classified as **Normal**, **Bacterial Pneumonia**, and **Viral Pneumonia**, indicating the **True Positives (TP)** for each category.

**MobileNetV2**

![image](https://github.com/m3mentomor1/Pneumonia_Detection_with_Lightweight-CNN-Models/assets/95956735/4ad21386-68e0-4f84-8190-5b10b30d9e9c)

- Bacterial Pneumonia: 167 out of 172 cases correctly predicted
- Viral Pneumonia: 124 out of 148 cases correctly predicted
- Normal: 145 out of 164 cases correctly predicted
<br>

**ShuffleNetV2**
<br>
![image](https://github.com/m3mentomor1/Pneumonia_Detection_with_Lightweight-CNN-Models/assets/95956735/d5e98fdd-1e44-4034-b8a6-7c3b1300d2cb)

- Bacterial Pneumonia: 163 out of 172 cases correctly predicted
- Viral Pneumonia: 124 out of 148 cases correctly predicted
- Normal: 141 out of 164 cases correctly predicted
<br>

**SqueezeNet**

![image](https://github.com/m3mentomor1/Pneumonia_Detection_with_Lightweight-CNN-Models/assets/95956735/a5c3634b-d16a-4b93-b8d8-204fdb439a81)

- Bacterial Pneumonia: 60 out of 172 cases correctly predicted
- Viral Pneumonia: 62 out of 148 cases correctly predicted
- Normal: 152 out of 164 cases correctly predicted
<br><br><br>
#### **➜ Classification Report**

**MobileNetV2**
|                     | Precision | Recall | F1-Score |
|---------------------|-----------|--------|----------|
| Bacterial Pneumonia | 0.87      | 0.97   | 0.92     |
| Viral Pneumonia     | 0.97      | 0.88   | 0.93     |
| Normal              | 0.87      | 0.84   | 0.85     |
<br>

**ShuffleNetV2**
|                     | Precision | Recall | F1-Score |
|---------------------|-----------|--------|----------|
| Bacterial Pneumonia | 0.88      | 0.95   | 0.91     |
| Viral Pneumonia     | 0.97      | 0.86   | 0.91     |
| Normal              | 0.82      | 0.84   | 0.83     |
<br>

**SqueezeNet**
|                     | Precision | Recall | F1-Score |
|---------------------|-----------|--------|----------|
| Bacterial Pneumonia | 0.82      | 0.35   | 0.49     |
| Viral Pneumonia     | 0.46      | 0.93   | 0.61     |
| Normal              | 0.79      | 0.42   | 0.55     |

MobileNetV2 demonstrates strong precision and recall scores across all categories, indicating balanced performance in correctly identifying positive cases (true positives) & effectively avoiding false positives & false negatives. ShuffleNetV2 also exhibits respectable precision & recall scores, particularly for the bacterial & viral pneumonia categories. However, SqueezeNet's classification report reveals lower precision & recall scores, especially for bacterial pneumonia, suggesting challenges in accurately identifying positive cases for this category.
<br><br><br>
#### **➜ Computational Efficiency**
| Model         | Model Size (MB) | Average Inference Speed (ms/image) |
|---------------|------------------|-----------------------------------|
| MobileNetV2   | 8.72             | 9.30                              |
| ShuffleNetV2  | 4.95             | 9.18                              |
| SqueezeNet    | 2.77             | 5.79                              |

The model sizes were determined from the file sizes after training, and the average inference speed was calculated by predicting 10 chest X-ray images from the test set and averaging the time taken to predict a single image in milliseconds.

As observed in the table, MobileNetV2 strikes a balance between model size and inference speed. ShuffleNetV2, while smaller in size compared to MobileNetV2, exhibits a slightly faster average inference speed. On the other hand, SqueezeNet distinguishes itself with its compact model size and relatively faster average inference speed. While MobileNetV2 and ShuffleNetV2 offer a balance between model size and speed, SqueezeNet prioritizes compactness, resulting in faster inference speeds at the expense model performance.

<br>Access the overall evaluation here: 
- [model_training-evaluation.ipynb](https://github.com/m3mentomor1/Pneumonia_Detection_with_Lightweight-CNN-Models/blob/main/model_training-evaluation.ipynb)
- [model_prediction.ipynb](https://github.com/m3mentomor1/Pneumonia_Detection_with_Lightweight-CNN-Models/blob/main/model_prediction.ipynb)
<br><br>
##

### 🛠️ V. Use this repository

**1. Clone this repository.**

   Run this command in your terminal: 
   ```
   git clone https://github.com/m3mentomor1/Pneumonia_Detection_with_Lightweight-CNN-Models.git
   ```
(Optional: You can also ```Fork``` this repository.)

<br>

**2. Go to the repository's main directory.**

   Run this command in your terminal: 
   ```
   cd Pneumonia_Detection_with_Lightweight-CNN-Models
   ```
<br>

##

### 🚀 VI. Model Deployment

![model-deploy](https://github.com/m3mentomor1/Pneumonia_Detection_with_Lightweight-CNN-Models/assets/95956735/db02a287-6a4a-42ea-a162-d9429de88136)

The models are deployed on [**Streamlit**](https://streamlit.io/). Click this [***link***](https://pneumoniadetectionwithlightweight-cnn-models-6sgynwffygezemyf8.streamlit.app/) to access the app.
<br><br>
##

### 📄 VII. License

👉 [Project License](https://github.com/m3mentomor1/Pneumonia_Detection_with_Lightweight-CNN-Models/blob/main/LICENSE)







