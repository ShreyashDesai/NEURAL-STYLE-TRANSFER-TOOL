import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import models, transforms
from PIL import Image
import matplotlib.pyplot as plt
import os

# --------------------------
# IMAGE LOADING + TRANSFORM
# --------------------------
def load_image(image_path, max_size=512):
    image = Image.open(image_path).convert('RGB')

    size = max(max(image.size), max_size)
    transform = transforms.Compose([
        transforms.Resize((512, 512)),
        transforms.ToTensor()
    ])

    image = transform(image).unsqueeze(0)
    return image


# --------------------------
# VGG19 FEATURE EXTRACTION
# --------------------------
class VGGFeatures(nn.Module):
    def __init__(self):
        super(VGGFeatures, self).__init__()
        vgg = models.vgg19(weights=models.VGG19_Weights.DEFAULT).features

        self.layers = {
            '0': 'conv1_1',
            '5': 'conv2_1',
            '10': 'conv3_1',
            '19': 'conv4_1',
            '21': 'conv4_2',  # content layer
            '28': 'conv5_1'
        }

        self.model = vgg[:29]

    def forward(self, x):
        features = {}
        for name, layer in self.model._modules.items():
            x = layer(x)
            if name in self.layers:
                features[self.layers[name]] = x
        return features


# --------------------------
# GRAM MATRIX (STYLE CALC)
# --------------------------
def gram_matrix(tensor):
    b, c, h, w = tensor.size()
    features = tensor.view(c, h * w)
    gram = torch.mm(features, features.t())
    return gram


# --------------------------
# MAIN STYLE TRANSFER FUNCTION
# --------------------------
def style_transfer(content_path, style_path, output_path, steps=300, lr=0.003):

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    content = load_image(content_path).to(device)
    style = load_image(style_path).to(device)

    target = content.clone().requires_grad_(True)

    model = VGGFeatures().to(device)

    optimizer = optim.Adam([target], lr=lr)

    for step in range(steps):
        target_features = model(target)
        content_features = model(content)
        style_features = model(style)

        # Content loss
        content_loss = torch.mean(
            (target_features['conv4_2'] - content_features['conv4_2']) ** 2
        )

        # Style loss
        style_loss = 0
        for layer in ['conv1_1', 'conv2_1', 'conv3_1', 'conv4_1', 'conv5_1']:
            target_gram = gram_matrix(target_features[layer])
            style_gram = gram_matrix(style_features[layer])
            style_loss += torch.mean((target_gram - style_gram) ** 2)

        total_loss = content_loss + 1e4 * style_loss

        optimizer.zero_grad()
        total_loss.backward()
        optimizer.step()

        if step % 50 == 0:
            print(f"STEP {step}/{steps}   LOSS: {total_loss.item():.4f}")

    # SAVE OUTPUT
    save_image = target.squeeze().detach().cpu()
    save_image = transforms.ToPILImage()(save_image)
    save_image.save(output_path)

    print(f"\nOutput saved to: {output_path}")


# --------------------------
# RUN DIRECTLY (EXAMPLE COMMAND)
# --------------------------
if __name__ == "__main__":
    content = "images/content/content.jpg"
    style = "images/style/style.jpg"
    output = "images/output/styled.jpg"

    style_transfer(content, style, output)

