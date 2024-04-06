import streamlit as st
from PIL import Image
import torch
import torchvision.transforms as transforms
import numpy as np
import io
import requests
from collections import OrderedDict
from io import BytesIO

# Define class names
class_names = ['Bacterial Pneumonia', 'Viral Pneumonia', 'Normal']

# Function to load model
def load_model(model_url):
    response = requests.get(model_url)
    if response.status_code == 200:
        st.write("Model loaded successfully")
        model_bytes = BytesIO(response.content)
        model = torch.load(model_bytes, map_location=torch.device('cpu'))
        st.write(f"Model type: {type(model)}")
        if isinstance(model, OrderedDict):
            model = extract_model_from_ordered_dict(model)
            if model is not None:
                return model
            else:
                st.write("Failed to extract model from OrderedDict")
                return None
        elif hasattr(model, 'eval'):
            model.eval()
            return model
        else:
            return None
    else:
        st.write("Failed to load model")
        return None

# Function to extract model from OrderedDict
def extract_model_from_ordered_dict(ordered_dict):
    for key in ordered_dict:
        if hasattr(ordered_dict[key], 'eval'):
            model = ordered_dict[key]
            model.eval()
            return model
    return None

# Function to preprocess image
def preprocess_image(image):
    # Convert to RGB if image mode is not RGB
    if image.mode != 'RGB':
        image = image.convert('RGB')
        
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])
    image = transform(image).unsqueeze(0)
    return image

# Function for prediction
def predict_image(image, model):
    with torch.no_grad():
        outputs = model(image)
        _, predicted = torch.max(outputs, 1)
        confidence = torch.nn.functional.softmax(outputs, dim=1)[0] * 100
    return predicted.item(), confidence.numpy()

# Streamlit app
st.title('Pneumonia Image Classification')

# Model selection
model_name = st.selectbox('Select Model', ['MobileNet-V2', 'ShuffleNet-V2', 'SqueezeNet 1.1', 'ResNet-18', 'EfficientNet-B0'])

# Load selected model
if model_name == 'MobileNet-V2':
    # Replace 'MOBILENET_V2_MODEL_DOWNLOAD_LINK' with the direct download link to the MobileNet-V2 model file
    model_url = 'https://github.com/m3mentomor1/Pneumonia_Detection_with_Lightweight-CNN-Models/raw/main/Models/mobilenetv2_model.pth'
    model = load_model(model_url)
elif model_name == 'ShuffleNet-V2':
    # Replace 'SHUFFLENET_V2_MODEL_DOWNLOAD_LINK' with the direct download link to the ShuffleNet-V2 model file
    model_url = 'https://github.com/m3mentomor1/Pneumonia_Detection_with_Lightweight-CNN-Models/raw/main/Models/shufflenetv2_model.pth'
    model = load_model(model_url)
elif model_name == 'SqueezeNet 1.1':
    # Replace 'SQUEEZENET_1_1_MODEL_DOWNLOAD_LINK' with the direct download link to the SqueezeNet 1.1 model file
    model_url = 'https://github.com/m3mentomor1/Pneumonia_Detection_with_Lightweight-CNN-Models/raw/main/Models/squeezenet1_1_model.pth'
    model = load_model(model_url)
elif model_name == 'ResNet-18':
    # Replace 'RESNET_18_MODEL_DOWNLOAD_LINK' with the direct download link to the ResNet-18 model file
    model_url_part1 = 'https://github.com/m3mentomor1/Pneumonia_Detection_with_Lightweight-CNN-Models/raw/main/Models/resnet18_model/resnet18_model.pth.part1'
    model_url_part2 = 'https://github.com/m3mentomor1/Pneumonia_Detection_with_Lightweight-CNN-Models/raw/main/Models/resnet18_model/resnet18_model.pth.part2'
    model = load_resnet18_model(model_url_part1, model_url_part2)
elif model_name == 'EfficientNet-B0':
    # Replace 'EFFICIENTNET_B0_MODEL_DOWNLOAD_LINK' with the direct download link to the EfficientNet-B0 model file
    model_url = 'https://github.com/m3mentomor1/Pneumonia_Detection_with_Lightweight-CNN-Models/raw/main/Models/efficientnetb0_model.pth'
    model = load_model(model_url)

# Function to load ResNet-18 model
def load_resnet18_model(model_url_part1, model_url_part2):
    response1 = requests.get(model_url_part1)
    response2 = requests.get(model_url_part2)
    if response1.status_code == 200 and response2.status_code == 200:
        st.write("ResNet-18 model files loaded successfully")
        model_bytes = BytesIO(response1.content + response2.content)
        model = torch.load(model_bytes, map_location=torch.device('cpu'))
        model.eval()
        return model
    else:
        st.write("Failed to load ResNet-18 model files")
        return None

# Image upload
uploaded_image = st.file_uploader("Choose an image...", type=['jpg', 'jpeg', 'png'])

if uploaded_image is not None:
    image = Image.open(uploaded_image)
    st.image(image, caption='Uploaded Image', use_column_width=True)
    transformed_image = preprocess_image(image)
    
    # Classify image
    if model is not None:
        prediction, confidence = predict_image(transformed_image, model)
        st.write(f"Prediction: {class_names[prediction]}")
        st.write(f"Confidence: {confidence[prediction]:.2f}%")
    else:
        st.write("Selected model not supported")
