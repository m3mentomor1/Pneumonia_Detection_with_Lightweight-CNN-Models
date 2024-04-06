import streamlit as st
import torch
import torchvision.transforms as transforms
from torchvision.models import mobilenet_v2, shufflenet_v2_x1_0, squeezenet1_1, resnet18
from PIL import Image
import matplotlib.pyplot as plt
from efficientnet_pytorch import EfficientNet
import torch.nn.functional as F
import requests
import io
import zipfile

# Define the base URL of the GitHub repository
base_url = "https://github.com/m3mentomor1/Pneumonia_Detection_with_Lightweight-CNN-Models/raw/main/Models/"

# Function to download and extract zip files
def download_and_extract_zip(url, target_path):
    response = requests.get(url)
    with zipfile.ZipFile(io.BytesIO(response.content), 'r') as zip_ref:
        zip_ref.extractall(target_path)

# Define the URLs of the zip files
resnet_model_zip_url_1 = base_url + "resnet18_model/resnet18_model.zip"
resnet_model_zip_url_2 = base_url + "resnet18_model/resnet18_model.zip"

# Define target paths for extracted files
target_path_1 = 'temp/resnet18_model'
target_path_2 = 'temp/resnet18_model'

# Download and extract the zip files
download_and_extract_zip(resnet_model_zip_url_1, target_path_1)
download_and_extract_zip(resnet_model_zip_url_2, target_path_2)

# Define the paths of the extracted model files
resnet_model_path_1 = 'temp/resnet18_model/resnet18_model.pth.part1'
resnet_model_path_2 = 'temp/resnet18_model/resnet18_model.pth.part2'

# Load the models from the extracted paths
resnet_model_1 = resnet18(pretrained=False)
resnet_model_1.fc = torch.nn.Linear(in_features=512, out_features=3, bias=True)
resnet_model_1.load_state_dict(torch.load(resnet_model_path_1, map_location=torch.device('cpu')))
resnet_model_1.eval()

resnet_model_2 = torch.load(resnet_model_path_2, map_location=torch.device('cpu'))
resnet_model_2.fc = torch.nn.Linear(in_features=512, out_features=3, bias=True)
resnet_model_2.eval()

# Define the transformations for input images
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

# Define the models dictionary
models = {
    "ResNet-18": [resnet_model_1, resnet_model_2]
}

# Header
st.title("Pneumonia Detection in Chest X-ray Images")

# Upload image
uploaded_image = st.file_uploader("Upload an image", type=["jpg", "jpeg", "png"])

# Model selection
selected_model = st.selectbox("Select Model", list(models.keys()))

if uploaded_image is not None:
    # Load the uploaded image
    test_image = Image.open(uploaded_image).convert('RGB')

    # Display the uploaded image
    st.image(test_image, caption='Uploaded Image', use_column_width=True)

    # Use the selected model for prediction
    model = models[selected_model]

    if selected_model == "ResNet-18":
        output = []
        for resnet_model in model:
            # Apply transformations to the test image
            input_image = transform(test_image).unsqueeze(0)

            # Make prediction
            with torch.no_grad():
                resnet_model.to(torch.device('cpu'))
                output.append(F.softmax(resnet_model(input_image), dim=1))
        probabilities = torch.mean(torch.stack(output), dim=0)
        confidence, predicted = torch.max(probabilities, 1)

    # Decode the predicted class
    class_names = ['Bacterial Pneumonia', 'Normal', 'Viral Pneumonia']
    predicted_class = class_names[predicted.item()]

    # Display the prediction
    st.write(f"Model: {selected_model}")
    st.write(f"Predicted Class: {predicted_class}")
    st.write(f"Confidence: {round(confidence.item(), 4)}")
