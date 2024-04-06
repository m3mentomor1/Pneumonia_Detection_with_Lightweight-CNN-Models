import streamlit as st
import torch
import torchvision.transforms as transforms
from torchvision.models import mobilenet_v2, shufflenet_v2_x1_0, squeezenet1_1, resnet18
from PIL import Image
import matplotlib.pyplot as plt
from efficientnet_pytorch import EfficientNet
import torch.nn.functional as F

# Define the paths of the saved models
mobilenet_model_path = "https://github.com/m3mentomor1/Pneumonia_Detection_with_Lightweight-CNN-Models/blob/main/Models/mobilenetv2_model.pth"
shufflenet_model_path = "https://github.com/m3mentomor1/Pneumonia_Detection_with_Lightweight-CNN-Models/blob/main/Models/shufflenetv2_model.pth"
squeezenet_model_path = "https://github.com/m3mentomor1/Pneumonia_Detection_with_Lightweight-CNN-Models/blob/main/Models/squeezenet1_1_model.pth"
resnet_model_path_1 = "https://github.com/m3mentomor1/Pneumonia_Detection_with_Lightweight-CNN-Models/blob/main/Models/resnet18_model/resnet18_model.pth.part1"
resnet_model_path_2 = "https://github.com/m3mentomor1/Pneumonia_Detection_with_Lightweight-CNN-Models/blob/main/Models/resnet18_model/resnet18_model.pth.part2"
efficient_net_model_path = "https://github.com/m3mentomor1/Pneumonia_Detection_with_Lightweight-CNN-Models/blob/main/Models/efficientnetb0_model.pth"

# Load the models from the saved paths
mobilenet_model = mobilenet_v2(pretrained=True)
mobilenet_model.classifier[1] = torch.nn.Linear(in_features=1280, out_features=3, bias=True)
mobilenet_model.load_state_dict(torch.load(mobilenet_model_path, map_location=torch.device('cpu')))
mobilenet_model.eval()

shufflenet_model = shufflenet_v2_x1_0(pretrained=True)
shufflenet_model.fc = torch.nn.Linear(in_features=1024, out_features=3, bias=True)
shufflenet_model.load_state_dict(torch.load(shufflenet_model_path, map_location=torch.device('cpu')))
shufflenet_model.eval()

squeezenet_model = squeezenet1_1(pretrained=True)
squeezenet_model.classifier[1] = torch.nn.Conv2d(512, 3, kernel_size=(1, 1), stride=(1, 1))
squeezenet_model.load_state_dict(torch.load(squeezenet_model_path, map_location=torch.device('cpu')))
squeezenet_model.eval()

resnet_model_1 = resnet18(pretrained=True)
resnet_model_1.fc = torch.nn.Linear(in_features=512, out_features=3, bias=True)
resnet_model_1.load_state_dict(torch.load(resnet_model_path_1, map_location=torch.device('cpu')))
resnet_model_2 = torch.load(resnet_model_path_2, map_location=torch.device('cpu'))
resnet_model_2.fc = torch.nn.Linear(in_features=512, out_features=3, bias=True)
resnet_model_1.eval()
resnet_model_2.eval()

efficient_net_model = EfficientNet.from_name('efficientnet-b0')
efficient_net_model._fc = torch.nn.Linear(in_features=1280, out_features=3, bias=True)
efficient_net_model.load_state_dict(torch.load(efficient_net_model_path, map_location=torch.device('cpu')))
efficient_net_model.eval()

# Define the transformations for input images
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

# Define the models dictionary
models = {
    "MobileNet-V2": mobilenet_model,
    "ShuffleNet-V2": shufflenet_model,
    "SqueezeNet 1.1": squeezenet_model,
    "ResNet-18": [resnet_model_1, resnet_model_2],
    "EfficientNet-B0": efficient_net_model
}

# Header
st.title("Pneumonia Detection in Chest X-ray Images")

# Upload image
uploaded_image = st.file_uploader("Upload an image", type=["jpg", "jpeg", "png"])

if uploaded_image is not None:
    # Load the uploaded image
    test_image = Image.open(uploaded_image).convert('RGB')

    # Display the uploaded image
    st.image(test_image, caption='Uploaded Image', use_column_width=True)

    # Iterate through each model and make predictions
    for model_name, model in models.items():
        if model_name == "ResNet-18":
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
        else:
            # Apply transformations to the test image
            input_image = transform(test_image).unsqueeze(0)

            # Make prediction
            with torch.no_grad():
                model.to(torch.device('cpu'))
                output = model(input_image)
                probabilities = F.softmax(output, dim=1)
                confidence, predicted = torch.max(probabilities, 1)

        # Decode the predicted class
        class_names = ['Bacterial Pneumonia', 'Normal', 'Viral Pneumonia']
        predicted_class = class_names[predicted.item()]

        # Display the prediction
        st.write(f"Model: {model_name}")
        st.write(f"Predicted Class: {predicted_class}")
        st.write(f"Confidence: {round(confidence.item(), 4)}")
