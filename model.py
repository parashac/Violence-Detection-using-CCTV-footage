import os
import cv2
import numpy as np

IMG_SIZE = 224
FRAMES = 30

def load_clip(folder_path):
    frames = []
    frame_files = sorted(os.listdir(folder_path))

    for fname in frame_files[:FRAMES]:
        img = cv2.imread(os.path.join(folder_path, fname))
        img = cv2.resize(img, (IMG_SIZE, IMG_SIZE))
        img = img / 255.0
        frames.append(img)

    return np.array(frames)

def load_dataset(root_dir):
    X, y = [], []

    class_map = {
        "non-violence": 0,
        "violence": 1
    }

    for class_name, label in class_map.items():
        class_path = os.path.join(root_dir, class_name)

        for clip in os.listdir(class_path):
            clip_path = os.path.join(class_path, clip)

            if not os.path.isdir(clip_path):
                continue

            frames = load_clip(clip_path)
            if frames.shape[0] != FRAMES:
                continue

            X.append(frames)
            y.append(label)

    return np.array(X), np.array(y)

from sklearn.model_selection import train_test_split
from tensorflow.keras.utils import to_categorical

X, y = load_dataset(
    "D:/Major project/dataset/converted-frames"
)

y = to_categorical(y, 2)

X_train, X_val, y_train, y_val = train_test_split(
    X, y,
    test_size=0.2,
    random_state=42,
    stratify=y
)

import tensorflow as tf
from tensorflow.keras import layers, models

base_cnn = tf.keras.applications.MobileNetV2(
    input_shape=(IMG_SIZE, IMG_SIZE, 3),
    include_top=False,
    weights="imagenet"
)

base_cnn.trainable = False

cnn_model = models.Sequential([
    base_cnn,
    layers.GlobalAveragePooling2D()
])

model = models.Sequential([
    layers.TimeDistributed(
        cnn_model,
        input_shape=(FRAMES, IMG_SIZE, IMG_SIZE, 3)
    ),

    layers.LSTM(64),

    layers.Dense(64, activation='relu'),
    layers.Dropout(0.4),

    layers.Dense(2, activation='softmax')
])

model.compile(
    optimizer=tf.keras.optimizers.Adam(1e-4),
    loss='categorical_crossentropy',
    metrics=['accuracy']
)

model.summary()

from tensorflow.keras.callbacks import (
    ModelCheckpoint,
    EarlyStopping,
    ReduceLROnPlateau
)

checkpoint = ModelCheckpoint(
    "violence_mobilenet_lstm_best.h5",
    monitor="val_loss",
    save_best_only=True,
    verbose=1
)

early_stop = EarlyStopping(
    monitor="val_loss",
    patience=6,
    restore_best_weights=True
)

reduce_lr = ReduceLROnPlateau(
    monitor="val_loss",
    factor=0.5,
    patience=3,
    verbose=1
)
history = model.fit(
    X_train, y_train,
    validation_data=(X_val, y_val),
    epochs=30,
    batch_size=4,
    callbacks=[checkpoint, early_stop, reduce_lr]
)
