# GAN-Guard: A Honeypot Password Generator

A project by the Tech Society at SEC. This repository contains the source code for GAN-Guard, a Generative Adversarial Network (GAN) trained to generate realistic, human-like passwords for use in cybersecurity honeypot systems.

---

## 📌 Overview

In cybersecurity, a **honeypot** is a decoy system used to attract and trap attackers. GAN-Guard is designed to create the "bait" for these traps. Instead of generating random, machine-like passwords, our AI learns the deep patterns from real-world password datasets to produce passwords that look authentically human-made. When an attacker attempts to use one of these generated passwords on a decoy account, it triggers an immediate security alert, allowing organizations to detect and respond to threats proactively.

## ✨ Features

- **Data Pipeline:** A complete data processing pipeline to clean, tokenize, and prepare massive password datasets for training.
- **GAN Model:** A sophisticated Generative Adversarial Network built with PyTorch, specifically designed for sequential data like text.
- **Training Module:** A robust script to train the GAN, including checkpointing and progress monitoring.
- **Password Generation Tool:** A user-friendly command-line interface (`run.py`) to generate passwords from a trained model.
- **Performance Evaluation Tool:** A quantitative analysis script (`evaluate.py`) to measure the quality of the generated passwords based on Uniqueness, Novelty, and Strength.

---

## 🚀 Getting Started

Follow these instructions to set up and run the project locally.

### Prerequisites

- Python 3.8+
- Git

### Installation & Setup

1.  **Clone the repository:**
    ```bash
    git clone [https://github.com/Tech-Society-SEC/Gan-guard.git](https://github.com/Tech-Society-SEC/Gan-guard.git)
    cd Gan-guard
    ```

2.  **Set up a Python virtual environment:**
    ```bash
    # For Windows
    python -m venv .venv
    .\.venv\Scripts\Activate.ps1

    # For macOS/Linux
    python3 -m venv .venv
    source .venv/bin/activate
    ```

3.  **Install the required dependencies:**
    ```bash
    pip install torch torchvision torchaudio zxcvbn-python
    ```

### How to Run

1.  **Preprocess the Data:**
    *(Note: You must have a `rockyou.txt` file in the project's root directory for this step).*
    ```bash
    python preprocess.py
    ```

2.  **Train the Model:**
    ```bash
    python train.py
    ```

3.  **Generate Passwords:**
    *(Replace `generator_epoch_X.pth` with your trained model file).*
    ```bash
    python run.py generator_epoch_X.pth --num 50
    ```

---

## 🤝 Our Team & Contributions

This project is a collaborative effort. We follow a feature-branch workflow, and all contributions are made via Pull Requests.

| Team Member        | GitHub Handle | Responsibilities                               |
| ------------------ | ------------- | ---------------------------------------------- |
| **BARATHRAJ K** | `@KBarathraj`  | **Project Lead**, Data Pipeline, Project Structure |
| **RISHON ANAND** | `@Rishon100`   | GAN Model Architecture (Generator & Discriminator) |
| **LOGESH B** | `@LogeshBalaji`   | Model Training and Evaluation Scripts            |

##  Sprint Log & Weekly Progress

This section tracks our progress on a weekly basis, aligning with our project sprints.

### **Project Foundation**

- **Goal:** Establish the core components of the project.
- **Pull Requests:**
  - `[#1] Feature: Data Processing Pipeline` - Implemented the foundational scripts to process and prepare the dataset.
  - `[#2] Feature: GAN Model Architecture` - Defined the core PyTorch models for the Generator and Discriminator.
  - `[#3] Feature: Initial Training Script` - Created the main script to handle the model training loop.
- **Outcome:** Successfully established a complete, runnable foundation for the project. All core components are in place for initial training experiments.
