🎨 NEURAL STYLE TRANSFER TOOL
--
CodTech IT Solutions – Internship Project (Artificial Intelligence)
--

Author: Shreyash Nhanu Desai
--
Intern ID: CT04DR1291
--
Domain: Artificial Intelligence
--
Duration: 4 Weeks
--
Mentor: Neela Santosh
--

--
📘 Project Overview

Neural Style Transfer (NST) is a deep learning technique that blends the content of one image with the artistic style of another.
This project uses Convolutional Neural Networks (CNNs) and the pretrained VGG19 model to recreate an image that looks like a photograph painted in the style of famous artworks.

This project demonstrates:

Computer vision

Image feature extraction

Deep learning optimization

🚀 Features

✔️ Apply any artistic style to any photograph
✔️ Uses pretrained VGG19 for feature extraction
✔️ Supports multiple style images
✔️ Automatically saves generated output
✔️ Beginner-friendly Python script + notebook
✔️ Clean project structure

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
│   └── content.jpg
│── style/
│   └── style.jpg
│── results/
│   └── output.png
│── requirements.txt
└── README.md

🧰 Installation & Setup Guide

Follow these steps carefully — this setup is designed so even a complete beginner can run the project easily.

🪜 Step 1 — Install Git

Git is required to clone the repository.

🔽 Download Git
👉 https://git-scm.com/downloads

Check installation:

git --version

🪜 Step 2 — Install Python

Download Python 3.10+
👉 https://www.python.org/downloads/

⚠ Important:
On the installer screen, check this box:
✔ Add Python to PATH

Verify installation:

python --version
pip --version

🪜 Step 3 — Clone the Repository
git clone https://github.com/ShreyashDesai/NEURAL-STYLE-TRANSFER-TOOL.git
cd NEURAL-STYLE-TRANSFER-TOOL

🪜 Step 4 — Install Required Libraries

Install all dependencies:

pip install -r requirements.txt


or install manually:

pip install torch torchvision pillow matplotlib

🪜 Step 5 — Run the Project
▶ Option 1: Run Jupyter Notebook
jupyter notebook style_transfer.ipynb

▶ Option 2: Run the Python Script
python neural_style_transfer.py


The final styled image will be saved here:

results/output.png

🧠 How Neural Style Transfer Works

NST separates an image into two key components:

🟦 Content Representation

Shapes

Edges

Structure of objects

🟧 Style Representation

Brush strokes

Texture

Color distribution

The neural network computes:

Content Loss: Keep original structure

Style Loss: Match artistic patterns

Total Loss: Content + Style

The output image is iteratively updated using gradient descent until the desired style is achieved.

🖼 Example Output
<img width="1801" height="610" alt="Screenshot 2025-11-28 073040" src="https://github.com/user-attachments/assets/2e1eaa3c-e481-48e2-a1bc-3072ae9e18a0" />
![output](https://github.com/user-attachments/assets/1c9a0134-576b-438f-bb5b-b1c3b66f5eda)


📧 Contact

Author: Shreyash Nhanu Desai
📩 Email: sheyashsn.desai@gmail.com

🔗 GitHub: https://github.com/ShreyashDesai

🔗 LinkedIn: https://www.linkedin.com/in/shreyash-desai-a13730384/

🏁 Acknowledgements

Thanks to CodTech IT Solutions and my mentor Neela Santosh for continuous support and guidance during this AI internship.
