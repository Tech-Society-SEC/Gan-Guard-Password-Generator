# GAN-Guard: A Honeypot Password Generator

A project by the Tech Society at SEC. This repository contains the source code for GAN-Guard, a Generative Adversarial Network (GAN) trained to generate realistic, human-like passwords for use in cybersecurity honeypot systems.

---

## 📌 Overview

In cybersecurity, a **honeypot** is a decoy system used to attract and trap attackers. GAN-Guard is designed to create the "bait" for these traps. Instead of generating random, machine-like passwords, our AI learns the deep patterns from real-world password datasets to produce passwords that look authentically human-made. When an attacker attempts to use one of these generated passwords on a decoy account, it triggers an immediate security alert, allowing organizations to detect and respond to threats proactively.

## ✨ Features

- **Advanced Data Curation:** A script (`curate_common.py`) to filter massive password lists, creating a focused dataset of only common, "humanlike" passwords.
- **Flexible Data Pipeline:** A processing script (`preprocess.py`) to tokenize and convert any text-based dataset into a PyTorch-ready tensor.
- **Stable WGAN-GP Model:** A state-of-the-art Wasserstein GAN with Gradient Penalty (`models.py`) to ensure stable training and prevent the mode collapse that simpler GANs face.
- **WGAN-GP Training Module:** A robust script (`train_wgan.py`) to train the stable model, including checkpointing and resume-from-epoch functionality.
- **Password Generation Tool:** A user-friendly CLI (`run.py`) to generate passwords from a trained model, with built-in filters to remove collapsed results (blanks/numbers).
- **Performance Evaluation Tool:** A quantitative analysis script (`evaluate.py`) to measure the quality of the generated passwords based on **Uniqueness**, **Novelty**, and **Strength**.

---

## 🎥 Demo Video

Watch the full demo of how our generated honeypot passwords plays the role here:
▶️ [Click to Watch on Google Drive](https://drive.google.com/file/d/123u5w7TCnE-tTqU2MQf3upalRfmNe5xq/view?usp=sharing)

---

## 🚀 Getting Started

Follow these instructions to set up and run the project locally.

### Prerequisites

- Python 3.8+
- Git
- An NVIDIA GPU with CUDA (recommended for training)

### Installation & Setup

1.  **Clone the repository:**
    ```bash
    git clone [https://github.com/Tech-Society-SEC/Gan-Guard-Password-Generator.git](https://github.com/Tech-Society-SEC/Gan-Guard-Password-Generator.git)
    cd Gan-Guard-Password-Generator
    ```

2.  **Set up a Python virtual environment:**
    ```bash
    # For macOS/Linux
    python3 -m venv .venv
    source .venv/bin/activate
    ```

3.  **Install the required dependencies:**
    ```bash
    pip install torch torchvision torchadudio zxcvbn-python
    ```

### How to Run (The Final "Humanlike" Model)

This is the process to train our final, stable, "humanlike" model.

1.  **Get Raw Data:**
    * Download the `rockyou.txt` password list and place it in the project's root directory.

2.  **Curate the "Common" Dataset:**
    * Run this script to filter the 14M passwords down to a smaller, focused list of only common, humanlike passwords (scores 0-1).
    ```bash
    python curate_common.py
    ```
    * This will create a new file: `rockyou_common_subset.txt`.

3.  **Preprocess the Data:**
    * Now, convert that new text file into a PyTorch tensor.
    ```bash
    python preprocess.py rockyou_common_subset.txt --output rockyou_common_subset_processed.pt
    ```

4.  **Train the WGAN-GP Model:**
    * This is our stable, advanced trainer.
    ```bash
    python train_wgan.py
    ```
    * This will save checkpoints (e.g., `generator_epoch_10.pth`) into the `models_wgan_common/` folder.

5.  **Generate Passwords:**
    * Use `run.py` to generate passwords from your "champion" model (e.g., Epoch 10).
    ```bash
    python run.py models_wgan_common/generator_epoch_10.pth --num 50
    ```

---

## 🛠️ Our 4-Stage Project Pipeline

This project followed a structured, 4-stage experimental pipeline.

### **Stage 1: Data Foundation & Curation**
- **Goal:** Create a high-quality, focused dataset.
- **Action:** We implemented `curate_common.py` to filter the 14M+ `rockyou.txt` dataset. We used `zxcvbn` to extract only the simple, "humanlike" passwords (scores 0-1), creating `rockyou_common_subset.txt`.
- **Outcome:** A small, fast, and high-quality dataset perfectly suited for our goal.

### **Stage 2: Initial Experiment & Failure Analysis**
- **Goal:** Establish a baseline and test a simple model.
- **Action:** We first trained a standard, simple GAN on our curated data.
- **Outcome:** A clear and important **failure**. The model was too simple, its gradients vanished, and it completely failed to learn (Loss stuck at `0.6931`). This proved we needed a more advanced architecture.

### **Stage 3: Advanced Model Development (WGAN-GP)**
- **Goal:** Solve the stability and "vanishing gradient" problem.
- **Action:** We replaced the simple GAN with a state-of-the-art **Wasserstein GAN with Gradient Penalty (WGAN-GP)**. This involved replacing the `Discriminator` with a `Critic` (no sigmoid) and implementing the advanced `train_wgan.py` script.
- **Outcome:** The WGAN-GP model trained perfectly. It was **stable, did not collapse, and showed clear learning** epoch after epoch.

### **Stage 4: Champion Model Selection & Application**
- **Goal:** Find the best-performing model and prove its quality.
- **Action:** We used `run.py` and `evaluate.py` to analyze the checkpoints from our successful WGAN training.
- **Outcome:** We identified **Epoch 10** as our **"champion" model**. It provided the perfect balance of visual quality (e.g., `pito977`, `buest`), high Uniqueness (92.5%), and high Novelty (88.1%), successfully achieving our project's goal.

---

## MODEL TRAINING
<img width="2521" height="1520" alt="image" src="https://github.com/user-attachments/assets/b5c2ec30-dca4-47e4-9b9a-7b1ccfa2a6f0" />

<img width="2521" height="1520" alt="image" src="https://github.com/user-attachments/assets/7920e34a-61bc-4848-8040-403fd4bb2543" />

## GENERATING PASSWORDS

<img width="2521" height="1520" alt="image" src="https://github.com/user-attachments/assets/d64d0f87-6fe1-4688-8067-494a1b964df0" />


## MODEL EVALUATION

<img width="2118" height="520" alt="image" src="https://github.com/user-attachments/assets/ab6b34d8-278e-45e3-939e-270d4e7bc29d" />



## 🤝 Our Team & Contributions

This project is a collaborative effort. We follow a feature-branch workflow, and all contributions are made via Pull Requests.

| Team Member | GitHub Handle | Responsibilities |
| :--- | :--- | :--- |
| **BARATHRAJ K** | `@KBarathraj` | **Project Lead**, Data Pipeline, WGAN-GP Implementation |
| **RISHON ANAND** | `@Rishon100` | GAN Model Architecture (Generator & Critic) |
| **LOGESH B** | `@LogeshBalaji` | Data Curation & Model Evaluation Scripts |
