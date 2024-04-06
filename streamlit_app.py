import streamlit as st
import torch
import torchvision.transforms as transforms
from torchvision.models import mobilenet_v2
from PIL import Image
import requests
import io

# Define the base URL of the GitHub repository
base_url = "https://github.com/m3mentomor1/Pneumonia_Detection_with_Lightweight-CNN-Models/raw/main/Models/"

# Define the path of the saved model
mobilenet_model_path = base_url + "mobilenetv2_model.pth"

# Load the model from the saved path
mobilenet_model = mobilenet_v2(pretrained=False)
mobilenet_model.classifier[1] = torch.nn.Linear(in_features=1280, out_features=3, bias=True)
mobilenet_model.load_state_dict(torch.load(io.BytesIO(requests.get(mobilenet_model_path).content), map_location=torch.device('cpu')))
mobilenet_model.eval()

# Define the transformations for input images
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

# Header
st.title("Pneumonia Detection in Chest X-ray Images")

# Upload image
uploaded_image = st.file_uploader("Upload an image", type=["jpg", "jpeg", "png"])

if uploaded_image is not None:
    # Load the uploaded image
    test_image = Image.open(uploaded_image).convert('RGB')

    # Display the uploaded image
    st.image(test_image, caption='Uploaded Image', use_column_width=True)

    # Apply transformations to the uploaded image
    input_image = transform(test_image).unsqueeze(0)

    # Make prediction
    with torch.no_grad():
        mobilenet_model.to(torch.device('cpu'))
        output = mobilenet_model(input_image)
        probabilities = torch.softmax(output, dim=1)
        confidence, predicted = torch.max(probabilities, 1)

    # Decode the predicted class
    class_names = ['Normal', 'Bacterial Pneumonia', 'Viral Pneumonia']
    predicted_class = class_names[predicted.item()]

    # Display the prediction
    st.write(f"Predicted Class: {predicted_class}")
    st.write(f"Confidence: {confidence.item()}")
