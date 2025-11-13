# Speech Emotion Recognition 🎤😄😡😢

[![Python](https://img.shields.io/badge/Python-3.7%2B-blue.svg)](https://www.python.org/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](./LICENSE)
[![Star](https://img.shields.io/github/stars/abhijit869/CodeAlpha_Project_Speech-Emotion-Recognition?style=social)](https://github.com/abhijit869/CodeAlpha_Project_Speech-Emotion-Recognition)
[![Issues](https://img.shields.io/github/issues/abhijit869/CodeAlpha_Project_Speech-Emotion-Recognition)](https://github.com/abhijit869/CodeAlpha_Project_Speech-Emotion-Recognition/issues)

Recognize human emotions from speech audio using cutting-edge Deep Learning and Signal Processing. This project automatically classifies emotions such as **Happy**, **Angry**, **Sad**, **Neutral**, and more from `.wav` files or microphone input.

---

## 🚀 Features

- 🎶 **Audio Input**: Supports `.wav` files and live microphone.
- 🧠 **Deep Learning Models**: Use CNN, LSTM, or hybrid architectures for best results.
- 📊 **Interactive Visualizations**: Confusion matrix, prediction plots, live stream graphs.
- 🗣 **Real-time Prediction** (Optional): See detected emotions instantly as you speak!
- 📦 **Modular Code**: Easy to extend for extra emotions, multilanguage, or other models.

---

## 📦 Installation

1. **Clone the repo:**
   ```bash
   git clone https://github.com/abhijit869/CodeAlpha_Project_Speech-Emotion-Recognition.git
   cd CodeAlpha_Project_Speech-Emotion-Recognition
   ```

2. **Create & activate a virtual environment (recommended):**
   ```bash
   python -m venv venv
   source venv/bin/activate  # On Windows: venv\Scripts\activate
   ```

3. **Install dependencies:**
   ```bash
   pip install -r requirements.txt
   ```

---

## 🧑‍💻 Usage

### 1. **Predict Emotion from WAV File**

```bash
python predict.py --input path/to/audio.wav
```

### 2. **Real-Time Microphone Emotion Recognition**

```bash
python mic_demo.py
```

### 3. **Train (or Retrain) Models**

```bash
python train.py --data_dir path/to/dataset
```

---

## 🎥 Demo

> **Add Demo GIFs or Screenshots here!**
>
> ![Demo GIF](assets/demo.gif)

---

## 🤓 Dataset

- Uses open datasets like [RAVDESS](https://zenodo.org/record/1188976#.YjzolmjTXIV), [CREMA-D](https://github.com/CheyneyComputerScience/CREMA-D), or your own `.wav` files.
- **Tip:** Place your dataset inside the `data/` folder and update config paths as needed.

---

## ⚙️ Configuration

You can modify parameters like model type, number of epochs, batch size, etc., in `config.py`.

---

## 📝 Contributing

Pull requests, suggestions, and questions are welcome!  
See [CONTRIBUTING.md](./CONTRIBUTING.md) for guidelines.

---

## 💬 Issues & Discussions

- [Open an Issue](https://github.com/abhijit869/CodeAlpha_Project_Speech-Emotion-Recognition/issues)
- [Start a Discussion](https://github.com/abhijit869/CodeAlpha_Project_Speech-Emotion-Recognition/discussions)

---

## 📜 License

This repo is licensed under the [MIT License](./LICENSE).

---

## 🙌 Acknowledgments

- [TensorFlow](https://www.tensorflow.org/)
- [Librosa](https://librosa.org/)
- [Scikit-learn](https://scikit-learn.org/)
- Audio datasets: RAVDESS, CREMA-D

---

### ⭐️ Star this repo if you find it useful!



