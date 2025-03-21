import os
import cv2
import numpy as np
import pandas as pd
import tensorflow as tf
from sklearn.model_selection import train_test_split

def load_images_from_folder(folder, label):
    images, labels = [], []
    for filename in os.listdir(folder):
        img_path = os.path.join(folder, filename)
        img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
        if img is not None:
            img = cv2.resize(img, (64, 64))
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
    model = tf.Module()
    model.conv1_weights = tf.Variable(tf.random.truncated_normal([3, 3, 1, 32], stddev=0.1))
    model.conv2_weights = tf.Variable(tf.random.truncated_normal([3, 3, 32, 64], stddev=0.1))
    model.fc1_weights = tf.Variable(tf.random.truncated_normal([16 * 16 * 64, 128], stddev=0.1))
    model.fc2_weights = tf.Variable(tf.random.truncated_normal([128, 4], stddev=0.1))

    model.conv1_bias = tf.Variable(tf.zeros([32]))
    model.conv2_bias = tf.Variable(tf.zeros([64]))
    model.fc1_bias = tf.Variable(tf.zeros([128]))
    model.fc2_bias = tf.Variable(tf.zeros([4]))

    def forward(x):
        x = tf.nn.conv2d(x, model.conv1_weights, strides=1, padding='SAME') + model.conv1_bias
        x = tf.nn.relu(x)
        x = tf.nn.max_pool2d(x, ksize=2, strides=2, padding='SAME')

        x = tf.nn.conv2d(x, model.conv2_weights, strides=1, padding='SAME') + model.conv2_bias
        x = tf.nn.relu(x)
        x = tf.nn.max_pool2d(x, ksize=2, strides=2, padding='SAME')

        x = tf.reshape(x, [-1, 16 * 16 * 64])
        x = tf.nn.relu(tf.matmul(x, model.fc1_weights) + model.fc1_bias)

        logits = tf.matmul(x, model.fc2_weights) + model.fc2_bias
        return logits

    model.forward = forward
    return model

def train_model():
    X, y, categories = prepare_dataset()
    X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=42)

    model = build_model()
    optimizer = tf.optimizers.Adam(learning_rate=0.001)

    for epoch in range(10):
        for i in range(0, len(X_train), 16):
            X_batch, y_batch = X_train[i:i+16], y_train[i:i+16]
            with tf.GradientTape() as tape:
                logits = model.forward(X_batch)
                loss = tf.reduce_mean(tf.nn.sparse_softmax_cross_entropy_with_logits(labels=y_batch, logits=logits))

            gradients = tape.gradient(loss, model.trainable_variables)
            optimizer.apply_gradients(zip(gradients, model.trainable_variables))

        print(f"Epoch {epoch + 1}/10, Loss: {loss.numpy():.4f}")

    tf.saved_model.save(model, "player_position_model_tf")
    return model, categories

def predict_test_images(model, categories):
    test_dir = "data/test"
    test_images, test_filenames = [], []

    for filename in sorted(os.listdir(test_dir), key=lambda x: int(x.split('.')[0])):
        img_path = os.path.join(test_dir, filename)
        img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
        if img is not None:
            img = cv2.resize(img, (64, 64)) / 255.0
            test_images.append(img.reshape(64, 64, 1))
            test_filenames.append(filename)

    test_images = np.array(test_images)
    predictions = model.forward(test_images)
    predicted_labels = [list(categories.keys())[np.argmax(pred)] for pred in predictions]

    df = pd.DataFrame(predicted_labels, columns=["Position"])
    df.to_csv("submission_tf.csv", index=False)
    print("Predictions saved to submission_tf.csv")

if __name__ == "__main__":
    model, categories = train_model()
    predict_test_images(model, categories)
