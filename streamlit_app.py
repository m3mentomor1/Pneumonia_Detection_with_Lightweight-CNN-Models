import streamlit as st
import torch
import torchvision.transforms as transforms
from torchvision.models import mobilenet_v2, shufflenet_v2_x1_0
from PIL import Image
import requests
import io

# Define the base URL of the GitHub repository
base_url = "https://github.com/m3mentomor1/Pneumonia_Detection_with_Lightweight-CNN-Models/raw/main/Models/"

# Define the paths of the saved models
mobilenet_model_path = base_url + "mobilenetv2_model.pth"
shufflenet_model_path = base_url + "shufflenetv2_model.pth"

# Header
st.title("Pneumonia Detection in Chest X-ray Images")

# Model selection
selected_model = st.selectbox("Select Model", ["MobileNetV2", "ShuffleNetV2"])

if selected_model == "MobileNetV2":
    # Load the MobileNetV2 model
    model = mobilenet_v2(pretrained=False)
    model.classifier[1] = torch.nn.Linear(in_features=1280, out_features=3, bias=True)
    model.load_state_dict(torch.load(io.BytesIO(requests.get(mobilenet_model_path).content), map_location=torch.device('cpu')))
elif selected_model == "ShuffleNetV2":
    # Load the ShuffleNetV2 model
    model = shufflenet_v2_x1_0(pretrained=False)
    model.fc = torch.nn.Linear(in_features=1024, out_features=3, bias=True)
    model.load_state_dict(torch.load(io.BytesIO(requests.get(shufflenet_model_path).content), map_location=torch.device('cpu')))
else:
    st.error("Invalid model selection")

# Define the transformations for input images
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

# Upload image
uploaded_image = st.file_uploader("Upload an image", type=["jpg", "jpeg", "png"])

if uploaded_image is not None:
    # Load the uploaded image
    test_image = Image.open(uploaded_image).convert('RGB')

    # Display the uploaded image
    st.image(test_image, caption='Uploaded Image', use_column_width=True)

    # Apply transformations to the test image
    input_image = transform(test_image).unsqueeze(0)

    # Make prediction
    with torch.no_grad():
        model.to(torch.device('cpu'))
        output = model(input_image)
        probabilities = torch.softmax(output, dim=1)
        confidence, predicted = torch.max(probabilities, 1)

    # Define the class names
    class_names = ['Bacterial Pneumonia', 'Normal', 'Viral Pneumonia']
    
    # Decode the predicted class
    predicted_class = class_names[predicted.item()]

    # Calculate confidence percentage
    confidence_percentage = round(confidence.item() * 100, 2)
    confidence_decimal = round(confidence.item(), 4)

    # Display the prediction
    st.write(f"Model: {selected_model}")
    st.write(f"Predicted Class: {predicted_class}")
    st.write(f"Confidence: {confidence_percentage}% ({confidence_decimal})")
