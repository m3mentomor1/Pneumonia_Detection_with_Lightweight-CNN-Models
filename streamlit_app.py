import streamlit as st
from PIL import Image
import torch
import torchvision.transforms as transforms
import numpy as np
import requests
from io import BytesIO
from collections import OrderedDict

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
            for key in model:
                if hasattr(model[key], 'eval'):
                    model = model[key]
                    model.eval()
                    return model
            # If no key with eval attribute is found, return None
            return None
        elif hasattr(model, 'eval'):
            model.eval()
            return model
        else:
            return None
    else:
        st.write("Failed to load model")
        return None

# Function to load ResNet-18 model
def load_resnet18_model(model_url_part1, model_url_part2):
    response_part1 = requests.get(model_url_part1)
    response_part2 = requests.get(model_url_part2)
    if response_part1.status_code == 200 and response_part2.status_code == 200:
        st.write("ResNet-18 model loaded successfully")
        model_bytes = BytesIO(response_part1.content + response_part2.content)
        model = torch.load(model_bytes, map_location=torch.device('cpu'))
        st.write(f"Model type: {type(model)}")
        if isinstance(model, OrderedDict):
            for key in model:
                if hasattr(model[key], 'eval'):
                    model = model[key]
                    model.eval()
                    return model
            # If no key with eval attribute is found, return None
            return None
        elif hasattr(model, 'eval'):
            model.eval()
            return model
        else:
            return None
    else:
        st.write("Failed to load ResNet-18 model")
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
model = None
if model_name == 'MobileNet-V2':
    model_url = 'https://github.com/m3mentomor1/Pneumonia_Detection_with_Lightweight-CNN-Models/raw/main/Models/mobilenetv2_model.pth'
    model = load_model(model_url)
elif model_name == 'ShuffleNet-V2':
    model_url = 'https://github.com/m3mentomor1/Pneumonia_Detection_with_Lightweight-CNN-Models/raw/main/Models/shufflenetv2_model.pth'
    model = load_model(model_url)
elif model_name == 'SqueezeNet 1.1':
    model_url = 'https://github.com/m3mentomor1/Pneumonia_Detection_with_Lightweight-CNN-Models/raw/main/Models/squeezenet1_1_model.pth'
    model = load_model(model_url)
elif model_name == 'ResNet-18':
    model_url_part1 = 'https://github.com/m3mentomor1/Pneumonia_Detection_with_Lightweight-CNN-Models/raw/main/Models/resnet18_model/resnet18_model.pth.part1'
    model_url_part2 = 'https://github.com/m3mentomor1/Pneumonia_Detection_with_Lightweight-CNN-Models/raw/main/Models/resnet18_model/resnet18_model.pth.part2'
    model = load_resnet18_model(model_url_part1, model_url_part2)
elif model_name == 'EfficientNet-B0':
    model_url = 'https://github.com/m3mentomor1/Pneumonia_Detection_with_Lightweight-CNN-Models/raw/main/Models/efficientnetb0_model.pth'
    model = load_model(model_url)

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
