import torch
import torchvision.transforms as transforms
from torchvision.datasets import ImageFolder
import os
from torch.utils.data import DataLoader
from sklearn.metrics import confusion_matrix, accuracy_score, precision_score, recall_score
import matplotlib.pyplot as plt
import seaborn as sns
from torchcam.methods import GradCAM
from torchcam.utils import overlay_mask
from torchvision.transforms.functional import to_pil_image
import torch.nn as nn
from torchvision import models
from PIL import Image
import torch.nn.functional as F
import warnings

# Suppress all warnings
warnings.filterwarnings("ignore")


# Set device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(device)

# Define data directories
data_dir = 'data'
scoliosis_dir = os.path.join(data_dir, 'scoliosis')
normal_dir = os.path.join(data_dir, 'normal')

# Define transformations
transform = transforms.Compose([
    transforms.Resize((224, 224)),  # Resize to match model input size
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),  # Add normalization
])

# Load the dataset
dataset = ImageFolder(root=data_dir, transform=transform)
train_size = int(0.8 * len(dataset))
test_size = len(dataset) - train_size
train_dataset, test_dataset = torch.utils.data.random_split(dataset, [train_size, test_size])

# Create DataLoaders
train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False)

# Modify classifier for each model
def modify_model(model, num_classes=2):
    if isinstance(model, models.ResNet):
        model.fc = nn.Linear(model.fc.in_features, num_classes)
    elif isinstance(model, models.DenseNet):
        model.classifier = nn.Linear(model.classifier.in_features, num_classes)
    elif isinstance(model, models.VisionTransformer):
        model.heads.head = nn.Linear(model.heads.head.in_features, num_classes)
    return model

# Load pre-trained models and modify them
resnet = models.resnet18(weights='IMAGENET1K_V1')
densenet = models.densenet121(weights='IMAGENET1K_V1')
vit = models.vit_b_16(weights='IMAGENET1K_V1')

resnet = modify_model(resnet).to(device)
densenet = modify_model(densenet).to(device)
vit = modify_model(vit).to(device)

def vit_gradcam(model, input_image, class_index, model_name, epoch):
    model.eval()
    input_tensor = input_image.unsqueeze(0).to(device)

    # Register hooks to get the output and gradients of the last attention layer
    activations = {}
    gradients = {}

    def save_activation(name):
        def hook(module, input, output):
            activations[name] = output
        return hook

    def save_gradient(name):
        def hook(module, grad_input, grad_output):
            gradients[name] = grad_output[0]
        return hook

    # Register hooks on the last attention layer
    last_attn = model.encoder.layers[-1].self_attention
    last_attn.register_forward_hook(save_activation('attn'))
    last_attn.register_full_backward_hook(save_gradient('attn'))

    # Forward pass
    output = model(input_tensor)

    # Clear existing gradients
    model.zero_grad()

    # Backward pass for the target class
    if output.shape[1] > 1:
        score = output[0, class_index]
    else:
        score = output[0]
    score.backward()

    # Get the gradients and attention weights
    attn_weights = activations['attn']
    attn_gradients = gradients['attn']

    # Handle tuple case for attention weights and gradients
    if isinstance(attn_weights, tuple):
        attn_weights = attn_weights[0]
    if isinstance(attn_gradients, tuple):
        attn_gradients = attn_gradients[0]

    print(f"Attention weights shape: {attn_weights.shape}")
    print(f"Attention gradients shape: {attn_gradients.shape}")

    # Calculate importance weights (gradients * activation)
    importances = torch.mean(attn_gradients, dim=1)  # Average over heads
    importances = importances.unsqueeze(1)

    # Weigh the attention maps by their importance
    weighted_attention = torch.sum(importances * attn_weights, dim=1)

    # Reshape the attention map to a square
    attention_size = int(weighted_attention.size(1) ** 0.5)
    attention_map = weighted_attention[:, 1:].view(1, 1, attention_size, attention_size)

    # Upsample to input image size
    attention_map = F.interpolate(attention_map,
                                  size=input_tensor.shape[2:],
                                  mode='bilinear',
                                  align_corners=False)

    # Normalize the attention map
    attention_map = attention_map.squeeze().cpu().detach().numpy()
    attention_map = (attention_map - attention_map.min()) / (attention_map.max() - attention_map.min())

    # Create heatmap
    heatmap = (attention_map * 255).astype('uint8')
    heatmap = Image.fromarray(heatmap).convert('RGB')

    # Overlay heatmap on original image
    original_img = to_pil_image(input_image.cpu())
    result = Image.blend(original_img, heatmap, 0.5)

    # Save the result
    result.save(f"{model_name}_gradcam_epoch_{epoch}.jpg")
    print(f"{model_name}_gradcam_epoch_{epoch}.jpg saved successfully.")

    
def apply_gradcam(model, input_image, target_layer, model_name,epoch):
    model.eval()
    cam_extractor = GradCAM(model, target_layer)
    
    # Get model output and Grad-CAM activation map
    output = model(input_image.unsqueeze(0).to(device))
    activation_map = cam_extractor(output.squeeze(0).argmax().item(), output)

    # Overlay CAM on image
    result = overlay_mask(to_pil_image(input_image), to_pil_image(activation_map[0].squeeze(0), mode='F'), alpha=0.5)
    
    # Save image
    result.save(f"{model_name}_{epoch}_gradcam.jpg")
    print(f"{model_name}_gradcam.jpg saved successfully.")

def generate_gradcam_images(model, input_image, model_name, epoch):
    if model_name.lower().startswith('vit'):
        vit_gradcam(model, input_image, 0, model_name, epoch)
    else:
        if isinstance(model, models.ResNet):
            target_layers = ['layer4']
        elif isinstance(model, models.DenseNet):
            target_layers = ['features.denseblock4.denselayer16']
        else:
            print(f"Unsupported model type: {type(model).__name__}")
            return
        
        for layer in target_layers:
            apply_gradcam(model, input_image, layer, model_name, epoch)

# Train model function
def train_model(model, criterion, optimizer, train_loader, num_epochs=10, model_name='model'):
    model = model.to(device)
    metrics = []

    for epoch in range(num_epochs):
        model.train()
        running_loss = 0.0
        correct = 0
        total = 0
        
        for i, (inputs, labels) in enumerate(train_loader):
            inputs, labels = inputs.to(device), labels.to(device)
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            running_loss += loss.item()
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

            # Generate Grad-CAM for the last batch of each epoch
            if not model_name.lower().startswith('vit'):
                if i == len(train_loader) - 1:
                    generate_gradcam_images(model, inputs[0], model_name, epoch + 1)
        
        train_acc = correct / total * 100
        print(f'Epoch [{epoch+1}/{num_epochs}], Loss: {running_loss/len(train_loader):.4f}, Accuracy: {train_acc:.2f}%')

        metrics.append((running_loss/len(train_loader), train_acc))
    
    torch.save(model.state_dict(), f"{model_name}.pth")
    print(f"{model_name} saved successfully.")
    
    return model, metrics

# Training parameters
criterion = nn.CrossEntropyLoss()
num_epochs = 10

# Train and save ResNet
optimizer_resnet = torch.optim.Adam(resnet.parameters(), lr=0.001)
trained_resnet, resnet_metrics = train_model(resnet, criterion, optimizer_resnet, train_loader, num_epochs, model_name='resnet18')

# Train and save DenseNet
optimizer_densenet = torch.optim.Adam(densenet.parameters(), lr=0.001)
trained_densenet, densenet_metrics = train_model(densenet, criterion, optimizer_densenet, train_loader, num_epochs, model_name='densenet121')

# Train and save ViT
optimizer_vit = torch.optim.Adam(vit.parameters(), lr=0.001)
trained_vit, vit_metrics = train_model(vit, criterion, optimizer_vit, train_loader, num_epochs, model_name='vit_b_16')

# Evaluate model function
def evaluate_model(model, test_loader, model_name):
    model.eval()
    true_labels = []
    pred_labels = []
    
    with torch.no_grad():
        for inputs, labels in test_loader:
            inputs, labels = inputs.to(device), labels.to(device)
            outputs = model(inputs)
            _, preds = torch.max(outputs, 1)
            true_labels.extend(labels.cpu().numpy())
            pred_labels.extend(preds.cpu().numpy())
    
    cm = confusion_matrix(true_labels, pred_labels)
    accuracy = accuracy_score(true_labels, pred_labels)
    precision = precision_score(true_labels, pred_labels, zero_division=0)
    recall = recall_score(true_labels, pred_labels)

    print(f'{model_name} - Accuracy: {accuracy:.2f}, Precision: {precision:.2f}, Recall: {recall:.2f}')
    plt.figure(figsize=(5, 5))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=['Normal', 'Scoliosis'], yticklabels=['Normal', 'Scoliosis'])
    plt.xlabel('Predicted')
    plt.ylabel('True')
    plt.title(f'Confusion Matrix for {model_name}')
    plt.savefig(f'{model_name}_confusion_matrix.png')
    plt.close()

# Evaluate each model
evaluate_model(trained_resnet, test_loader, 'ResNet')
evaluate_model(trained_densenet, test_loader, 'DenseNet')
evaluate_model(trained_vit, test_loader, 'ViT')

# Classify image function
def classify_image(model, image_path, transform):
    image = Image.open(image_path)
    image = transform(image).unsqueeze(0).to(device)
    
    model.eval()
    with torch.no_grad():
        outputs = model(image)
        _, predicted = torch.max(outputs, 1)
    
    return "Yes" if predicted.item() == 1 else "No"

# Example usage
xray_image_path = input("Enter the path of the X-ray image: ")
result = classify_image(trained_resnet, xray_image_path, transform)
print(f"Scoliosis detected: {result}")
