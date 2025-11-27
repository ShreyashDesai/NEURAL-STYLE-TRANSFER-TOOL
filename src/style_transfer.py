import torch
import torch.nn as nn
import torch.optim as optim
from PIL import Image
import torchvision.transforms as transforms
import torchvision.models as models
import matplotlib.pyplot as plt
import os

# Load image and resize
def load_image(image_path, max_size=512):
    image = Image.open(image_path).convert("RGB")
    size = max(image.size)
    if size > max_size:
        size = max_size

    transform = transforms.Compose([
        transforms.Resize((size, size)),
        transforms.ToTensor()
    ])

    image = transform(image).unsqueeze(0)
    return image.to(device)

# Convert tensor to displayable image
def im_convert(tensor):
    image = tensor.clone().detach().cpu().numpy().squeeze()
    image = image.transpose(1, 2, 0)
    return image

# Style loss using Gram matrix
def gram_matrix(tensor):
    b, c, h, w = tensor.size()
    tensor = tensor.view(c, h * w)
    gram = torch.mm(tensor, tensor.t())
    return gram

# Device configuration
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Load content and style images
content_path = "./images/content/content.jpg"
style_path = "./images/style/style.jpg"

content = load_image(content_path)
style = load_image(style_path)

# Load pretrained VGG19 model
vgg = models.vgg19(pretrained=True).features.to(device).eval()

# Freeze weights
for param in vgg.parameters():
    param.requires_grad = False

content_layers = ['conv4_2']
style_layers = ['conv1_1', 'conv2_1', 'conv3_1', 'conv4_1', 'conv5_1']

# Extract features
def get_features(image, model):
    features = {}
    x = image

    layer_map = {
        "0": "conv1_1", "5": "conv2_1", "10": "conv3_1",
        "19": "conv4_1", "21": "conv4_2", "28": "conv5_1"
    }

    for name, layer in model._modules.items():
        x = layer(x)
        if name in layer_map:
            features[layer_map[name]] = x

    return features

content_features = get_features(content, vgg)
style_features = get_features(style, vgg)

# Compute Gram matrices for style layers
style_grams = {layer: gram_matrix(style_features[layer]) for layer in style_layers}

# Create target image
target = content.clone().requires_grad_(True).to(device)

# Set weights
style_weight = 1e6
content_weight = 1

optimizer = optim.Adam([target], lr=0.003)
steps = 2000

print("Running style transfer...")

for i in range(1, steps+1):

    target_features = get_features(target, vgg)

    content_loss = torch.mean((target_features['conv4_2'] - content_features['conv4_2'])**2)

    style_loss = 0
    for layer in style_layers:
        target_feature = target_features[layer]
        target_gram = gram_matrix(target_feature)
        style_gram = style_grams[laye]()
