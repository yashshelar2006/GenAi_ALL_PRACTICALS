# 🧑‍🎨 EXP7 — StyleGAN Face Generation

> A Generative AI application that uses a trained StyleGAN-based generator to synthesize realistic human face images from latent representations.

---

## 📌 Overview

This project demonstrates **image generation using Generative Adversarial Networks (GANs)**.

A trained generator model is used to transform a randomly sampled latent vector into a synthetic human face. The generated image is then displayed through a simple web interface built using **Flask, HTML, and CSS**.

The experiment focuses on understanding how a trained generative model can learn visual patterns and generate new images that were not directly present in the input.

---

## 🎯 Objective

The main objectives of this experiment are:

- Understand the basic concept of **Generative Adversarial Networks**
- Explore the **StyleGAN architecture**
- Understand the role of the **generator**
- Learn how latent vectors can be converted into images
- Load and use a trained PyTorch model
- Integrate a deep learning model with a web application
- Generate synthetic human faces using a trained generator

---

## 🧠 What is StyleGAN?

**StyleGAN** is a generative model architecture developed by NVIDIA for high-quality image synthesis.

Unlike a traditional GAN where a latent vector is directly provided to the generator, StyleGAN introduces a style-based generation process.

Different parts of the network can influence different visual characteristics of the generated image.

For example:

```text
Latent Representation
        ↓
Style Mapping / Generator
        ↓
Learned Visual Features
        ↓
Synthetic Face
```

The important idea is that the generator has learned patterns from training data and can use those learned patterns to create new images.

---

## ⚙️ How This Project Works

The application follows the following pipeline:

```text
        Random Latent Vector
                │
                ▼
      Trained StyleGAN Model
                │
                ▼
        Generator Network
                │
                ▼
      Synthetic Face Image
                │
                ▼
          Flask Backend
                │
                ▼
          Web Interface
```

### Step-by-step

1. A latent vector is generated.
2. The latent representation is passed to the trained generator.
3. The generator transforms the latent representation into an image.
4. The generated image is converted into a displayable format.
5. Flask serves the generated image through the web application.
6. The user can generate a new face through the interface.

---

## ✨ Features

- 🎨 Synthetic human face generation
- 🧠 Trained deep learning generator
- ⚡ PyTorch-based inference
- 🌐 Flask web application
- 🖥️ Simple browser-based interface
- 🔄 Generate new images on demand
- 🍎 Supports Apple Silicon acceleration through MPS when available

---

## 🛠️ Tech Stack

| Technology | Purpose |
|------------|---------|
| Python | Core programming language |
| PyTorch | Deep learning model and inference |
| StyleGAN | Image generation architecture |
| Flask | Backend web framework |
| HTML/CSS | User interface |
| NumPy | Numerical operations |
| Pillow | Image processing |

---

## 📁 Project Structure

```text
EXP7/
│
├── model/
│   └── stylegan_trained.pth
│
├── app.py
├── index.html
├── requirements.txt
├── README.md
└── .gitignore
```

### File Description

**`app.py`**

Contains the Flask application and model inference logic.

**`index.html`**

Provides the frontend interface used to interact with the generator.

**`model/stylegan_trained.pth`**

Contains the trained PyTorch model weights used for image generation.

**`requirements.txt`**

Contains the Python dependencies required to run the project.

**`.gitignore`**

Prevents unnecessary files such as virtual environments, Python cache files, and macOS system files from being committed.

---

## 🚀 Installation & Setup

### 1. Clone the repository

```bash
git clone https://github.com/Kartikibansod/Generative-AI.git
cd Generative-AI/EXP7
```

### 2. Create a virtual environment

```bash
python3 -m venv venv
```

### 3. Activate the virtual environment

#### macOS / Linux

```bash
source venv/bin/activate
```

#### Windows

```bash
venv\Scripts\activate
```

### 4. Install dependencies

```bash
pip install -r requirements.txt
```

---

## ▶️ Run the Application

Start the Flask server:

```bash
python app.py
```

The terminal will display the local application address.

Open the displayed URL in your browser.

For example:

```text
http://127.0.0.1:5000
```

---

## 🖥️ Using the Application

Once the application is running:

```text
Open Browser
     ↓
Open Flask URL
     ↓
Click "Generate"
     ↓
Latent Vector
     ↓
StyleGAN Generator
     ↓
Generated Face
```

Each generation can produce a different synthetic face depending on the latent representation used.

---

## 🔬 Model

The project uses a trained PyTorch model stored as:

```text
model/stylegan_trained.pth
```

The `.pth` file contains the learned model parameters required during inference.

It is loaded by the Flask backend when the application starts.

```text
stylegan_trained.pth
        ↓
PyTorch
        ↓
Generator
        ↓
Latent Vector
        ↓
Generated Image
```

---

## 🧩 Generator vs. Latent Vector

These two concepts are important for understanding the project.

### Latent Vector

A latent vector is a numerical representation that provides the generator with information from which an image can be synthesized.

```text
Random numbers
      ↓
Latent Vector
      ↓
Generator
```

### Generator

The **generator is the neural network that actually creates the image**.

```text
Latent Vector
      ↓
Generator
      ↓
Image
```

In this project:

```text
Latent Vector → Trained Generator → Synthetic Face
```

---

## 📊 Output

The application produces a synthetic human face generated by the trained model.

The output is **not a photograph retrieved from a database**. It is generated by the neural network from its learned representation.

---

## 📚 Learning Outcomes

Through this experiment, I learned:

- Fundamentals of GAN-based image generation
- The concept of latent representations
- The role of a generator in GANs
- Basics of StyleGAN
- Loading PyTorch model weights
- Performing model inference
- Image processing using Pillow
- Connecting a deep learning model with Flask
- Creating a simple frontend for a generative AI application
- Running deep learning inference on Apple Silicon using MPS when available

---

## ⚠️ Limitations

This project is intended primarily as a learning experiment.

Possible limitations include:

- Generated image quality depends on the trained model
- The model is limited by the characteristics of its training data
- The application performs inference using a pre-trained model rather than training StyleGAN from scratch
- Generation speed depends on available hardware
- The current interface provides limited control over generated attributes

---

## 🔮 Future Improvements

Possible extensions include:

- Add user-controlled latent seeds
- Add image generation history
- Allow users to download generated images
- Add adjustable generation parameters
- Add latent-space interpolation
- Add side-by-side image comparison
- Improve the frontend design
- Deploy the application online
- Add more advanced StyleGAN controls

---

## 🧪 Experiment Summary

This experiment demonstrates the complete path from a **latent representation to a generated image** and shows how a trained generative AI model can be integrated into a web application.

```text
                GENERATIVE AI
                     │
                     ▼
              Latent Vector
                     │
                     ▼
             StyleGAN Generator
                     │
                     ▼
             Synthetic Face
                     │
                     ▼
                Flask App
                     │
                     ▼
              Web Browser
```

---

