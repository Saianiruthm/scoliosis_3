# fastapi_app.py
from fastapi import FastAPI, File, UploadFile
from fastapi.middleware.cors import CORSMiddleware
import torch
import torchvision.transforms as transforms
from PIL import Image
import io
from torchvision import models
import torch.nn as nn

app = FastAPI()

# Enable CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Define the transform
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])

# Load models
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def modify_model(model, num_classes=2):
    if isinstance(model, models.ResNet):
        model.fc = nn.Linear(model.fc.in_features, num_classes)
    elif isinstance(model, models.DenseNet):
        model.classifier = nn.Linear(model.classifier.in_features, num_classes)
    elif isinstance(model, models.VisionTransformer):
        model.heads.head = nn.Linear(model.heads.head.in_features, num_classes)
    return model

resnet = modify_model(models.resnet18(weights='IMAGENET1K_V1')).to(device)
densenet = modify_model(models.densenet121(weights='IMAGENET1K_V1')).to(device)
vit = modify_model(models.vit_b_16(weights='IMAGENET1K_V1')).to(device)

resnet.load_state_dict(torch.load('resnet18.pth', map_location=device))
densenet.load_state_dict(torch.load('densenet121.pth', map_location=device))
vit.load_state_dict(torch.load('vit_b_16.pth', map_location=device))

model_dict = {
    'resnet': resnet,
    'densenet': densenet,
    'vit': vit
}

@app.post("/predict")
async def predict(file: UploadFile = File(...), model_type: str = 'all'):
    contents = await file.read()
    image = Image.open(io.BytesIO(contents)).convert('RGB')
    image_tensor = transform(image).unsqueeze(0).to(device)

    results = {}

    if model_type == 'all':
        for name, model in model_dict.items():
            model.eval()
            with torch.no_grad():
                output = model(image_tensor)
                _, predicted = torch.max(output, 1)
                results[name] = "Yes" if predicted.item() == 1 else "No"
    elif model_type in model_dict:
        model = model_dict[model_type]
        model.eval()
        with torch.no_grad():
            output = model(image_tensor)
            _, predicted = torch.max(output, 1)
            results[model_type] = "Yes" if predicted.item() == 1 else "No"
    else:
        return {"error": "Invalid model type"}

    return results

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
