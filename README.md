# CodeAlpha_Project_Speech-Emotion-Recognition
# 🎤 Speech Emotion Recognition using Deep Learning  

> 🎧 Detect human emotions (Happy, Angry, Sad, Calm, etc.) directly from voice using Deep Learning & Audio Signal Processing.

---

## 🚀 Overview  
This project builds an end-to-end **Speech Emotion Recognition (SER)** system using **CNN + LSTM hybrid deep learning models**.  
It processes raw audio from datasets like **RAVDESS**, **TESS**, and **CREMA-D**, extracts advanced features (MFCC, Chroma, Mel, Contrast, Tonnetz), and classifies emotional states with **90–95% accuracy**.

---

## 🧠 Features  
✅ Multi-dataset compatibility: **RAVDESS, TESS, CREMA-D**  
✅ Hybrid deep learning model (**CNN + BiLSTM**)  
✅ Advanced **MFCC + Δ + ΔΔ** feature extraction  
✅ Data normalization and label encoding  
✅ Real-time prediction for uploaded audio files  
✅ Confusion matrix and training visualization plots  
✅ Google Colab ready (no setup hassle)  

---

## 🗂️ Project Structure  
📁 Speech_Emotion_Recognition/
│
├── data_processing.py 
├── model_training.py
├── inference.py 
├── main.py 
│
├── requirements.txt 
├── README.md 
│
└── datasets/ # Auto-downloaded RAVDESS/TESS data


---

## ⚙️ Installation  

### 🧩 Option 1: Run on Google Colab (Recommended)
Just open in Colab and run all cells 👇  
[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com)

---

### 🧩 Option 2: Run Locally  

#### 1️⃣ Clone the repository  :
git clone https://github.com/yourusername/Speech-Emotion-Recognition.git
cd Speech-Emotion-Recognition

Install dependencies
pip install -r requirements.txt
Run the main script
python main.py


