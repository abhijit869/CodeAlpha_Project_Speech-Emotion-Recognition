import pickle
import numpy as np
from tensorflow.keras.models import load_model
from data_processing import extract_features

class EmotionInference:
    def __init__(self, model_path='best_emotion_model.h5', scaler_path='scaler.pkl', le_path='label_encoder.pkl'):
        self.model = load_model(model_path)
        with open(scaler_path, 'rb') as f:
            self.scaler = pickle.load(f)
        with open(le_path, 'rb') as f:
            self.le = pickle.load(f)

    def predict_file(self, file_path):
        feat = extract_features(file_path)
        if feat is None:
            return None
        scaled = self.scaler.transform([feat])
        if len(self.model.input_shape) == 3:
            scaled = scaled.reshape(1, scaled.shape[1], 1)
        pred = self.model.predict(scaled)
        idx = int(np.argmax(pred, axis=1)[0])
        label = self.le.inverse_transform([idx])[0]
        conf = float(np.max(pred))
        probs = {self.le.classes_[i]: float(pred[0][i]) for i in range(len(self.le.classes_))}
        return {'label': label, 'confidence': conf, 'probs': probs}
