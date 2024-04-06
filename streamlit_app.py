import streamlit as st
import torch
import torchvision.transforms as transforms
from torchvision.models import mobilenet_v2
from PIL import Image
import requests
import io
import torch.nn.functional as F

# Define the base URL of the GitHub repository
base_url = "https://github.com/m3mentomor1/Pneumonia_Detection_with_Lightweight-CNN-Models/raw/main/Models/"

# Define the path of the saved model
mobilenet_model_path = base_url + "mobilenetv2_model.pth"

# Load the MobileNet-V2 model
mobilenet_model = mobilenet_v2(pretrained=False)
mobilenet_model.classifier[1] = torch.nn.Linear(in_features=1280, out_features=3, bias=True)

try:
    # Load the model's state dict from the provided path
    state_dict = torch.load(io.BytesIO(requests.get(mobilenet_model_path).content), map_location=torch.device('cpu'))
    mobilenet_model.load_state_dict(state_dict)
    st.success("MobileNet-V2 model loaded successfully!")
except Exception as e:
    st.error(f"Error loading MobileNet-V2 model: {e}")

# Header
st.title("Pneumonia Detection in Chest X-ray Images")

# Upload image
uploaded_image = st.file_uploader("Upload an image", type=["jpg", "jpeg", "png"])

# Define the transformations for input images
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

# Function to make predictions
def predict(image):
    # Ensure the image is not None
    if image is not None:
        # Load the uploaded image
        img = Image.open(image).convert('RGB')

        # Display the uploaded image
        st.image(img, caption='Uploaded Image', use_column_width=True)

        # Apply transformations to the test image
        input_image = transform(img).unsqueeze(0)

        # Make prediction
        with torch.no_grad():
            mobilenet_model.to(torch.device('cpu'))
            output = mobilenet_model(input_image)
            probabilities = F.softmax(output, dim=1)
            confidence, predicted = torch.max(probabilities, 1)

        # Decode the predicted class
        class_names = ['Bacterial Pneumonia', 'Normal', 'Viral Pneumonia']
        predicted_class = class_names[predicted.item()]

        # Display the prediction
        st.write(f"Predicted Class: {predicted_class}")
        st.write(f"Confidence: {round(confidence.item(), 4)}")

# Trigger prediction upon image upload
predict(uploaded_image)
