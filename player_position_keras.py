import os
import cv2
import numpy as np
import pandas as pd
from tensorflow import keras
from keras import layers
from sklearn.model_selection import train_test_split

def load_images_from_folder(folder, label):
    images = []
    labels = []
    for filename in os.listdir(folder):
        img_path = os.path.join(folder, filename)
        img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
        if img is not None:
            img = cv2.resize(img, (64, 64))  # Resize for consistency
            images.append(img)
            labels.append(label)
    return images, labels

def prepare_dataset():
    data_dir = "data/train"
    categories = {"Forward": 0, "Midfielder": 1, "Defender": 2, "Goalkeeper": 3}
    
    X, y = [], []
    for category, label in categories.items():
        folder = os.path.join(data_dir, category)
        images, labels = load_images_from_folder(folder, label)
        X.extend(images)
        y.extend(labels)
    
    X = np.array(X).reshape(-1, 64, 64, 1) / 255.0  # Normalize
    y = np.array(y)
    return X, y, categories

def build_model():
    model = keras.Sequential([
        layers.Conv2D(32, (3,3), activation='relu', input_shape=(64, 64, 1)),
        layers.MaxPooling2D((2,2)),
        layers.Conv2D(64, (3,3), activation='relu'),
        layers.MaxPooling2D((2,2)),
        layers.Flatten(),
        layers.Dense(128, activation='relu'),
        layers.Dense(4, activation='softmax')
    ])
    
    model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])
    return model

def train_model():
    X, y, categories = prepare_dataset()
    X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=42)
    
    model = build_model()
    model.fit(X_train, y_train, epochs=10, validation_data=(X_val, y_val), batch_size=16)
    model.save("player_position_model.h5")
    return model, categories

def predict_test_images(model, categories):
    test_dir = "data/test"
    test_images = []
    test_filenames = []
    
    for filename in sorted(os.listdir(test_dir), key=lambda x: int(x.split('.')[0])):
        img_path = os.path.join(test_dir, filename)
        img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
        if img is not None:
            img = cv2.resize(img, (64, 64)) / 255.0
            test_images.append(img.reshape(64, 64, 1))
            test_filenames.append(filename)
    
    test_images = np.array(test_images)
    predictions = model.predict(test_images)
    predicted_labels = [list(categories.keys())[np.argmax(pred)] for pred in predictions]
    
    df = pd.DataFrame(predicted_labels, columns=["Position"])
    df.to_csv("submission_keras.csv", index=False)
    print("Predictions saved to submission.csv")

if __name__ == "__main__":
    model, categories = train_model()
    predict_test_images(model, categories)
