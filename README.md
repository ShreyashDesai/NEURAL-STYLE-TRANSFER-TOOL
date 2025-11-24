🎨 NEURAL STYLE TRANSFER TOOL

Company: CodTech IT Solutions
Name: Shreyash Nhanu Desai
Intern ID: CT04DR1291
Domain: Artificial Intelligence
Duration: 4 Weeks
Mentor: Neela Santosh

📘 Project Overview

The Neural Style Transfer (NST) project uses Deep Learning to apply the artistic style of one image (e.g., a painting) onto another image (e.g., a photograph).

This technique uses Convolutional Neural Networks (CNNs) and pretrained models such as VGG19 to merge content and style into a stylized output image.

This project demonstrates the power of Computer Vision, Feature Extraction, and Optimization within modern AI systems.

🚀 Features

🖼️ Apply artistic style to any photograph

⚙️ Uses pretrained VGG19 deep learning model

🔁 Supports multiple style images

💾 Saves output images

💻 Implemented in a clean and simple Python Notebook

📊 Includes visual examples and comparisons

🛠️ Technologies Used
Category	Technology
Language	Python
Libraries	torch, torchvision, PIL, matplotlib
Model	Pretrained VGG19
Algorithm	Gatys' Neural Style Transfer
🖥️ How Neural Style Transfer Works

NST is based on the concept of separating and recombining:

Content representation — shapes & structures of the content image

Style representation — brush strokes, colors & textures of the style image

Using a loss function:

Content Loss measures similarity to the content image

Style Loss uses Gram matrices to measure texture similarity

The model uses gradient descent to iteratively update pixels of the output image.

📂 Project Structure
Neural-Style-Transfer/
│── style_transfer.ipynb
│── content/
│     └── content.jpg
│── style/
│     └── style.jpg
│── results/
│     └── output.png
│── README.md
└── requirements.txt

💻 How to Run
🪜 Step 1 — Install Python

Download from:
👉 https://www.python.org/downloads/

Check:

python --version
pip --version

🪜 Step 2 — Install Dependencies
pip install torch torchvision pillow matplotlib

🪜 Step 3 — Run the Notebook
jupyter notebook style_transfer.ipynb


Or run the script version:

python neural_style_transfer.py

🧩 Example Output

Content Image:
A regular photograph.

Style Image:
A famous painting.

Result:
A stylized image combining the content of the photograph with the artistic style of the painting.

<p align="center"> <img src="https://github.com/user-attachments/assets/cbf5cc21-682d-49d5-945b-f70e17b89c73" width="80%" /> </p>
🧠 Model Information

Model Used:
📌 VGG19 (pretrained on ImageNet)
Used for extracting both:

High-level content features

Low-level style features

NST uses only feature maps — the model weights remain frozen.

👨‍💻 Author

Shreyash Nhanu Desai
Intern at CodTech IT Solutions

📧 Email: sheyashsn.desai@gmail.com

🔗 GitHub: https://github.com/ShreyashDesai

🔗 LinkedIn: https://www.linkedin.com/in/shreyash-desai-a13730384/

🏁 Acknowledgements

I sincerely thank CodTech IT Solutions and my mentor Neela Santosh for their guidance throughout this internship and for providing me the opportunity to work on this exciting deep learning project.
