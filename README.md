🎨 NEURAL STYLE TRANSFER TOOL
CodTech IT Solutions Internship Project

Author: Shreyash Nhanu Desai
Intern ID: CT04DR1291
Domain: Artificial Intelligence
Duration: 4 Weeks
Mentor: Neela Santosh

📘 Project Overview

The Neural Style Transfer (NST) project applies the artistic style of one image (painting) onto another image (photograph).
It uses Deep Learning, Convolutional Neural Networks (CNNs), and the pretrained VGG19 model to merge:

Content (structure of the image)

Style (textures, brush strokes, colors)

This project demonstrates the power of computer vision, feature extraction, and neural optimization.

🚀 Features

🖼️ Apply any artistic style to any photograph

⚡ Uses pretrained VGG19 model

🔁 Supports multiple style images

💾 Automatically saves output images

💻 Clean, beginner-friendly Python Notebook

📊 Includes visual comparisons of content, style & output

🛠️ Technologies Used
Category	Technology
Language	Python
Libraries	torch, torchvision, Pillow, matplotlib
Model	Pretrained VGG19
Algorithm	Gatys' Neural Style Transfer
📂 Project Structure
Neural-Style-Transfer/
│── style_transfer.ipynb
│── neural_style_transfer.py
│── content/
│     └── content.jpg
│── style/
│     └── style.jpg
│── results/
│     └── output.png
│── README.md
└── requirements.txt

🧰 Installation & Setup Guide
Follow these steps exactly — even a complete beginner can do it.
🪜 Step 1 — Install Git

Git is required to clone the repository.

🔽 Download Git

👉 https://git-scm.com/downloads

✔ Check installation

Open Command Prompt / PowerShell and run:

git --version


If it shows a version, Git is installed correctly.

🪜 Step 2 — Install Python

Download Python 3.10+ from:
👉 https://www.python.org/downloads/

⚠ IMPORTANT
On the installer screen, CHECK the option:

✔ Add Python to PATH

Verify installation:
python --version
pip --version

🪜 Step 3 — Clone the Repository

Run this command:

git clone https://github.com/ShreyashDesai/Neural-Style-Transfer.git


Then enter the folder:

cd Neural-Style-Transfer

🪜 Step 4 — Install Required Libraries

Install all dependencies using:

pip install -r requirements.txt


If you want to install manually:

pip install torch torchvision pillow matplotlib

🪜 Step 5 — Run the Project
▶ Option 1: Run the Notebook
jupyter notebook style_transfer.ipynb

▶ Option 2: Run the Python Script
python neural_style_transfer.py


Your output image will be saved inside:

results/output.png

🧠 How Neural Style Transfer Works

NST separates and recombines:

🟦 Content Representation

Shapes, edges, and structure of the main image.

🟧 Style Representation

Textures, color patterns, brush strokes from the style image.

The model computes:

Content Loss → Keep structure similar

Style Loss → Match color & texture patterns

Total Loss = Content Loss + Style Loss

The output image is updated using gradient descent until the style is transferred.

🖼 Example Output
<img width="1801" height="610" alt="Image" src="https://github.com/user-attachments/assets/413eb1de-bd9e-4603-9af6-d5ceb21e011c" />
📧 Contact

Author: Shreyash Nhanu Desai
📩 Email: sheyashsn.desai@gmail.com

🔗 GitHub: https://github.com/ShreyashDesai

🔗 LinkedIn: https://www.linkedin.com/in/shreyash-desai-a13730384/

🏁 Acknowledgements

Special thanks to CodTech IT Solutions and my mentor Neela Santosh for providing guidance and support throughout this internship.
