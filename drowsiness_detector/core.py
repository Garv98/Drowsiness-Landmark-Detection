import os
import sys
import json
import logging
import subprocess
import cv2
import numpy as np
import tensorflow as tf
tf.config.run_functions_eagerly(True)
from tensorflow.keras.models import Sequential
from tensorflow.keras.models import load_model
from tensorflow.keras.layers import (Dense, Flatten, Conv2D, MaxPooling2D, Dropout, BatchNormalization)
from tensorflow.keras.preprocessing.image import ImageDataGenerator
import matplotlib.pyplot as plt

def _lazy_mp():
    import sys, os
    stderr_saved = sys.stderr
    sys.stderr = open(os.devnull, 'w')
    import mediapipe as mp
    sys.stderr.close()
    sys.stderr = stderr_saved
    return mp

os.environ['GLOG_minloglevel'] = '3'
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
os.environ['MEDIAPIPE_DISABLE_GPU'] = '1'
from absl import logging as absl_logging
absl_logging.set_verbosity(absl_logging.ERROR)

from zipfile import ZipFile
from datetime import datetime

from sklearn.preprocessing import LabelBinarizer
from sklearn.model_selection import train_test_split
from dotenv import load_dotenv

load_dotenv()

datasets = ['rakibuleceruet/drowsiness-prediction-dataset', 'adinishad/prediction-images']
api_token = {"username": "garvagarwalla", "key": os.getenv('KAGGLE_API')}
user_path = os.path.expanduser('~')

def build_drowsiness_model():
    model = Sequential([
        Conv2D(16, 3, activation='relu', input_shape=(145,145,3)),
        BatchNormalization(), MaxPooling2D(), Dropout(0.1),
        Conv2D(32, 5, activation='relu'),
        BatchNormalization(), MaxPooling2D(), Dropout(0.1),
        Conv2D(64, 10, activation='relu'),
        BatchNormalization(), MaxPooling2D(), Dropout(0.1),
        Conv2D(128, 12, activation='relu'),
        BatchNormalization(), Flatten(),
        Dense(512, activation='relu'), Dropout(0.1),
        Dense(1, activation='sigmoid')
    ])
    model.compile(
        optimizer='adam',
        loss='binary_crossentropy',
        metrics=['accuracy'],
        run_eagerly=True
    )
    return model

def setup_kaggle():
    kaggle_dir = os.path.join(user_path, '.kaggle')
    os.makedirs(kaggle_dir, exist_ok=True)
    with open(os.path.join(kaggle_dir, 'kaggle.json'), 'w') as f:
        json.dump(api_token, f)

def download_datasets(datasets):
    for ds in datasets:
        subprocess.call(['kaggle', 'datasets', 'download', '-d', ds, '-p', './Data'])
        zip_file = os.path.join('./Data', ds.split('/')[-1] + '.zip')
        with ZipFile(zip_file, 'r') as z:
            z.extractall(os.path.join('./Data', ds.split('/')[-1]))

def setup_dirs(categories):
    for cat in categories:
        os.makedirs(os.path.join('Data', 'landmarks', cat), exist_ok=True)
    os.makedirs('Models', exist_ok=True)
    os.makedirs('Logs', exist_ok=True)

mp = _lazy_mp()
mp_drawing = mp.solutions.drawing_utils
mp_facemesh = mp.solutions.face_mesh
mp_face_detection = mp.solutions.face_detection
face_detector = mp_face_detection.FaceDetection(model_selection=1, min_detection_confidence=0.5)

chosen_left_eye_idxs  = [362, 385, 387, 263, 373, 380]
chosen_right_eye_idxs = [33, 160, 158, 133, 153, 144]
all_idxs = chosen_left_eye_idxs + chosen_right_eye_idxs

def add_landmarks(name, img, category, landmarks, save_img=False):
    h, w = img.shape[:2]
    output = img.copy()
    spec = mp_drawing.DrawingSpec(thickness=1, circle_radius=2, color=(255, 255, 255))
    mp_drawing.draw_landmarks(
        image=output,
        landmark_list=landmarks,
        connections=mp_facemesh.FACEMESH_TESSELATION,
        connection_drawing_spec=spec
    )
    for idx in all_idxs:
        x = int(landmarks.landmark[idx].x * w)
        y = int(landmarks.landmark[idx].y * h)
        cv2.circle(output, (x, y), 3, (255, 255, 255), -1)
    if save_img:
        cv2.imwrite(os.path.join('Data', 'landmarks', category, name), output)
    return output

def process_image(image, category, name, save_img=True):
    IMG_SIZE = 145
    if image is None:
        logging.warning(f"{name}: image read failed")
        return None
    h, w = image.shape[:2]
    
    results = face_detector.process(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
    if not results.detections:
        logging.warning(f"{name}: no face detected")
        return None
    bbox = results.detections[0].location_data.relative_bounding_box
    x1 = max(0, int(bbox.xmin * w))
    y1 = max(0, int(bbox.ymin * h))
    x2 = min(w, x1 + int(bbox.width * w))
    y2 = min(h, y1 + int(bbox.height * h))
    
    margin = 0.1
    dx, dy = int((x2 - x1) * margin), int((y2 - y1) * margin)
    x1, y1 = max(0, x1 - dx), max(0, y1 - dy)
    x2, y2 = min(w, x2 + dx), min(h, y2 + dy)
    roi = image[y1:y2, x1:x2]
    if min(roi.shape[:2]) < 50:
        logging.warning(f"{name}: ROI too small {roi.shape}")
        return None
    
    with mp_facemesh.FaceMesh(static_image_mode=True, max_num_faces=1, min_detection_confidence=0.5) as mesh:
        mesh_res = mesh.process(cv2.cvtColor(roi, cv2.COLOR_BGR2RGB))
    if not mesh_res.multi_face_landmarks:
        logging.warning(f"{name}: no landmarks found")
        return None
    
    annotated = add_landmarks(name, roi.copy(), category, mesh_res.multi_face_landmarks[0], save_img)
    return cv2.resize(annotated, (IMG_SIZE, IMG_SIZE))

def process_dataset(root_dir, categories):
    setup_dirs(categories)
    data = []
    for category in categories:
        folder = os.path.join(root_dir, category)
        if not os.path.isdir(folder):
            logging.warning(f"Missing folder: {folder}")
            continue
        for filename in os.listdir(folder):
            try:
                img_path = os.path.join(folder, filename)
                img = cv2.imread(img_path)
                result = process_image(img, category, filename, save_img=True)
                if result is not None:
                    data.append([result, categories.index(category)])
            except Exception as e:
                logging.warning(f"{filename}: {e}")
    return data

def load_landmarks(categories):
    data = []
    IMG_SIZE = 145
    for category in categories:
        folder = os.path.join('Data', 'landmarks', category)
        if not os.path.isdir(folder):
            logging.warning(f"Missing landmarks folder: {folder}")
            continue
        for img_file in os.listdir(folder):
            path = os.path.join(folder, img_file)
            img = cv2.imread(path)
            if img is None:
                logging.warning(f"Could not read landmark image: {img_file}")
                continue
            data.append([cv2.resize(img, (IMG_SIZE, IMG_SIZE)), categories.index(category)])
    return data

def setup_training_data(data,img_size=(145,145),batch_size=32,validation_split=0.2):
    train_datagen = ImageDataGenerator(
        rescale=1./255,
        zoom_range=0.2,
        rotation_range=30,
        horizontal_flip=True,
        validation_split=validation_split
    )
    val_datagen = ImageDataGenerator(
        rescale=1./255,
        validation_split=validation_split
    )
    train_gen = train_datagen.flow_from_directory(
        data,
        target_size=img_size,
        batch_size=batch_size,
        class_mode='binary',
        subset='training',
        shuffle=True
    )
    val_gen = val_datagen.flow_from_directory(
        data,
        target_size=img_size,
        batch_size=batch_size,
        class_mode='binary',
        subset='validation',
        shuffle=False
    )
    return train_gen, val_gen


def load_saved_model(load_last=True):
    model_dir = 'Models'
    if load_last and os.path.isdir(model_dir):
        h5_files = sorted(f for f in os.listdir(model_dir) if f.lower().endswith('.h5'))
        if h5_files:
            latest = h5_files[-1]
            model_path = os.path.join(model_dir, latest)
            print("Model path:", model_path)
            # load only architecture + weights, skip optimizer
            model = load_model(model_path, compile=False)
            model.compile(
                optimizer='adam',
                loss='binary_crossentropy',
                metrics=['accuracy'],
                run_eagerly=True
            )
            return model
    return build_drowsiness_model()

def train_model(model, train_gen, test_gen, epochs=5):
    history = model.fit(
        train_gen,
        validation_data=test_gen,
        epochs=epochs
    )
    
    timestamp = datetime.now().strftime('%Y%m%d%H%M%S')
    out_path = os.path.join('Models', f'model_{timestamp}.h5')
    model.save(out_path)
    logging.info(f"Model saved to {out_path}")
    
    plt.figure(figsize=(12, 4))

    plt.subplot(1, 2, 1)
    plt.plot(history.history['accuracy'], label='Train Accuracy')
    plt.plot(history.history['val_accuracy'], label='Validation Accuracy')
    plt.title('Model Accuracy')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy')
    plt.legend()
    plt.grid(True)

    plt.subplot(1, 2, 2)
    plt.plot(history.history['loss'], label='Train Loss')
    plt.plot(history.history['val_loss'], label='Validation Loss')
    plt.title('Model Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()
    plt.grid(True)

    plt.tight_layout()
    plt.savefig("training_plot.png")
    plt.show()

    return history

def evaluate_model(model, test_gen):
    loss, acc = model.evaluate(test_gen)
    logging.info(f"Evaluation -> Loss: {loss:.4f}, Accuracy: {acc:.4f}")
    return loss, acc

def predict(model, image):
    proc = process_image(image, '', 'live', save_img=False)
    if proc is None:
        return None
    proc = proc.astype('float32') / 255.0
    proc = proc.reshape(1, 145, 145, 3)
    p = model.predict(proc)[0][0]
    return 1 if p >= 0.5 else 0