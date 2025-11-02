import os
import numpy as np
import librosa
import pickle
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from tensorflow.keras.utils import to_categorical
from sklearn.preprocessing import StandardScaler

def extract_features(file_path, duration=3, offset=0.5):
    try:
        audio, sr = librosa.load(file_path, duration=duration, offset=offset)
        mfcc = librosa.feature.mfcc(y=audio, sr=sr, n_mfcc=40)
        delta = librosa.feature.delta(mfcc)
        delta2 = librosa.feature.delta(mfcc, order=2)
        feat = np.hstack((np.mean(mfcc.T, axis=0), np.mean(delta.T, axis=0), np.mean(delta2.T, axis=0)))
        return feat
    except Exception:
        return None

def load_ravdess(path):
    X, y = [], []
    emo_map = {'01':'neutral','02':'calm','03':'happy','04':'sad','05':'angry','06':'fearful','07':'disgust','08':'surprised'}
    for root, _, files in os.walk(path):
        for f in files:
            if f.endswith('.wav'):
                parts = f.split('-')
                if len(parts) >= 3:
                    code = parts[2]
                    label = emo_map.get(code)
                    if label:
                        feat = extract_features(os.path.join(root, f))
                        if feat is not None:
                            X.append(feat)
                            y.append(label)
    return np.array(X), np.array(y)

def prepare_data(data_path='datasets/RAVDESS', test_size=0.2, random_state=42):
    X, y = load_ravdess(data_path)
    le = LabelEncoder()
    y_enc = le.fit_transform(y)
    num_classes = len(le.classes_)
    y_cat = to_categorical(y_enc, num_classes)
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    X_train, X_test, y_train, y_test = train_test_split(X_scaled, y_cat, test_size=test_size, random_state=random_state, stratify=y_enc)
    with open('scaler.pkl','wb') as f:
        pickle.dump(scaler, f)
    with open('label_encoder.pkl','wb') as f:
        pickle.dump(le, f)
    return X_train, X_test, y_train, y_test, scaler, le, num_classes
