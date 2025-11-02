import argparse
import os
from data_processing import prepare_data
from model_training import create_cnn_bilstm, train_model
from inference import EmotionInference
import numpy as np

def run_train(data_path, model_type):
    X_train, X_test, y_train, y_test, scaler, le, num_classes = prepare_data(data_path)
    X_train_r = X_train.reshape(X_train.shape[0], X_train.shape[1], 1)
    X_test_r = X_test.reshape(X_test.shape[0], X_test.shape[1], 1)
    model = create_cnn_bilstm((X_train.shape[1],1), num_classes)
    history = train_model(model, X_train_r, y_train, X_test_r, y_test)
    model.save('ravdess_emotion_model.h5')
    return history

def run_serve(example_file):
    inf = EmotionInference(model_path='best_emotion_model.h5')
    print(inf.predict_file(example_file))

if __name__ == '__main__':
    p = argparse.ArgumentParser()
    p.add_argument('--mode', choices=['train','serve'], default='train')
    p.add_argument('--data_path', default='datasets/RAVDESS')
    p.add_argument('--example', default='example.wav')
    args = p.parse_args()
    if args.mode == 'train':
        run_train(args.data_path, 'CNN-BiLSTM')
    else:
        run_serve(args.example)
