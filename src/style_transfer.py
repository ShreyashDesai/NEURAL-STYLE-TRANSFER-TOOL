import torch
import torch.nn as nn
import torch.optim as optim
from PIL import Image
import torchvision.transforms as transforms
import torchvision.models as models
import os

# -----------------------------
# PATHS (WORKS WITH YOUR FOLDER)
# -----------------------------
CONTENT_IMG_PATH = "images/content/content.jpg"
STYLE_IMG_PATH = "images/style/style.jpg"
OUTPUT_IMG_PATH = "images/output/output.jpg"


# -----------------------------
# LOAD IMAGE FUNCTION
# -----------------------------
def load_image(image_path, max_size=512):
    image = Image.open(image_path).convert("RGB")

    size = max(image.size)
    if size > max_size:
        scale = max_size / size
        new_w = int(image.size[0] * scale)
        new_h = int(image.size[1] * scale)
        image = image.resize((new_w, new_h))

    transform = transforms.ToTensor()
    image = transform(image).unsqueeze(0)
    return image


# -----------------------------
# LOAD VGG MODEL
# -----------------------------
def get_vgg_layers():
    vgg = models.vgg19(weights=models.VGG19_Weights.DEFAULT).features
    for param in vgg.parameters():
        param.requires_grad = False
    return vgg


# -----------------------------
# STYLE TRANSFER (MAIN)
# -----------------------------
def run_style_transfer():

    # Load content & style images
    content = load_image(CONTENT_IMG_PATH)
    style = load_image(STYLE_IMG_PATH)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    content = content.to(device)
    style = style.to(device)

    vgg = get_vgg_layers().to(device)

    # Layers to use
    style_layers = [0, 5, 10, 19, 28]
    content_layer = 21

    # Extract features
    def get_features(image):
        features = []
        x = image
        for i, layer in enumerate(vgg):
            x = layer(x)
            if i in style_layers or i == content_layer:
                features.append(x)
        return features

    style_features = get_features(style)
    content_features = get_features(content)

    # Gram matrix (for style)
    def gram_matrix(tensor):
        _, c, h, w = tensor.size()
        tensor = tensor.view(c, h * w)
        gram = torch.mm(tensor, tensor.t())
        return gram

    style_grams = [gram_matrix(feat) for feat in style_features]

    # Generated image
    generated = content.clone().requires_grad_(True)
    optimizer = optim.Adam([generated], lr=0.02)

    # Weights
    style_weight = 1e6
    content_weight = 1

    # Training loop
    for step in range(300):
        generated_features = get_features(generated)

        content_loss = torch.mean((generated_features[-1] - content_features[-1]) ** 2)

        style_loss = 0
        for gf, sg in zip(generated_features[:-1], style_grams):
            gm = gram_matrix(gf)
            style_loss += torch.mean((gm - sg) ** 2)

        total_loss = content_weight * content_loss + style_weight * style_loss

        optimizer.zero_grad()
        total_loss.backward()
        optimizer.step()

        if step % 50 == 0:
            print(f"Step {step}, Loss: {total_loss.item()}")

    # Save output
    generated = generated.cpu().clone().squeeze(0)
    generated = transforms.ToPILImage()(generated)
    generated.save(OUTPUT_IMG_PATH)

    print("\nStyle Transfer Completed!")
    print(f"Image saved at: {OUTPUT_IMG_PATH}")


# -----------------------------
# MAIN
# -----------------------------
if __name__ == "__main__":
    # Create output folder if not exists
    if not os.path.exists("images/output"):
        os.makedirs("images/output")

    run_style_transfer()
