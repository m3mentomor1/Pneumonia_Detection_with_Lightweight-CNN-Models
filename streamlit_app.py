import streamlit as st
import torch
import torchvision.transforms as transforms
from torchvision.models import mobilenet_v2, shufflenet_v2_x1_0, squeezenet1_1
from PIL import Image
import requests
import io
import time

# Define the base URL of the GitHub repository
base_url = "https://github.com/m3mentomor1/Pneumonia_Detection_with_Lightweight-CNN-Models/raw/main/Models/"

# Define the paths of the saved models
mobilenet_model_path = base_url + "mobilenetv2_model.pth"
shufflenet_model_path = base_url + "shufflenetv2_model.pth"
squeezenet_model_path = base_url + "squeezenet1_1_model.pth"

# Load the MobileNetV2 model
mobilenet_model = mobilenet_v2(pretrained=False)
mobilenet_model.classifier[1] = torch.nn.Linear(in_features=1280, out_features=3, bias=True)
mobilenet_model.load_state_dict(torch.load(io.BytesIO(requests.get(mobilenet_model_path).content), map_location=torch.device('cpu')))
mobilenet_model.eval()

# Load the ShuffleNetV2 model
shufflenet_model = shufflenet_v2_x1_0(pretrained=False)
shufflenet_model.fc = torch.nn.Linear(in_features=1024, out_features=3, bias=True)
shufflenet_model.load_state_dict(torch.load(io.BytesIO(requests.get(shufflenet_model_path).content), map_location=torch.device('cpu')))
shufflenet_model.eval()

# Load the SqueezeNet model
squeezenet_model = squeezenet1_1(pretrained=False)
squeezenet_model.classifier[1] = torch.nn.Conv2d(512, 3, kernel_size=(1, 1), stride=(1, 1))
squeezenet_model.load_state_dict(torch.load(io.BytesIO(requests.get(squeezenet_model_path).content), map_location=torch.device('cpu')))
squeezenet_model.eval()

# Define the transformations for input images
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

# Header
st.title("Pneumonia Detection in Chest X-ray Images")

# Model selection
selected_model = st.selectbox("Select a model", ["MobileNet-V2", "ShuffleNet-V2", "SqueezeNet"])

# Determine selected model
if selected_model == "MobileNet-V2":
    model = mobilenet_model
    model_name = "MobileNet-V2"
elif selected_model == "ShuffleNet-V2":
    model = shufflenet_model
    model_name = "ShuffleNet-V2"
elif selected_model == "SqueezeNet":
    model = squeezenet_model
    model_name = "SqueezeNet"
else:
    st.error("Invalid model selection")

# Upload image
uploaded_image = st.file_uploader("Upload a chest X-ray image", type=["jpg", "jpeg", "png"])

if uploaded_image is not None:
    # Load the uploaded image
    test_image = Image.open(uploaded_image).convert('RGB')

    # Display the uploaded image
    st.image(test_image, caption='Uploaded Image', use_column_width=True)

    # Apply transformations to the test image
    input_image = transform(test_image).unsqueeze(0)

    # Make prediction and measure inference time
    with torch.no_grad():
        model.to(torch.device('cpu'))
        start_time = time.time()
        output = model(input_image)
        inference_time_ms = (time.time() - start_time) * 1000  # Convert to milliseconds
        probabilities = torch.softmax(output, dim=1)
        confidence, predicted = torch.max(probabilities, 1)

    # Define the class names
    class_names = ['Bacterial Pneumonia', 'Normal', 'Viral Pneumonia']
    
    # Decode the predicted class
    predicted_class = class_names[predicted.item()]

    # Calculate confidence percentage
    confidence_percentage = round(confidence.item() * 100, 2)
    confidence_decimal = round(confidence.item(), 4)

    # Display the prediction and inference time
    st.write(f"Model: {model_name}")
    st.write(f"Predicted Class: {predicted_class}")
    st.write(f"Confidence: {confidence_percentage}% ({confidence_decimal})")
    st.write(f"Inference Time: {inference_time_ms:.2f} ms")
