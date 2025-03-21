# Player Position Classification using CNN

# PyTorch

This project implements a convolutional neural network (CNN) to classify football player positions (Forward, Midfielder, Defender, Goalkeeper) based on heatmap images. The model is built using PyTorch and follows a standard deep learning pipeline for image classification.

## Project Structure
```
player-position/
├── data/
│   ├── train/
│   │     ├── Forward/
│   │     ├── Midfielder/
│   │     ├── Defender/
│   │     └── Goalkeeper/
│   └── test/
├── player_position_pytorch.py
├── player_position_model.pth
└── submission_pytorch.csv
```

- **data/train/**: Contains training images categorized by player positions.
- **data/test/**: Contains test images for evaluation.
- **player_position_pytorch.py**: Main Python script to train the model and generate predictions.
- **submission.csv**: Output CSV file containing the predicted player positions for test images.

## Requirements
Ensure you have Python 3.10+ and the following packages installed:

```bash
pip install torch torchvision numpy pandas scikit-learn opencv-python
```

## Model Architecture
The CNN model consists of the following layers:

1. Two convolutional layers with ReLU activation and max pooling
2. Fully connected layer (128 neurons) with ReLU activation
3. Output layer with 4 neurons (one for each class)

### Model Definition:
```python
class PlayerPositionCNN(nn.Module):
    def __init__(self):
        super(PlayerPositionCNN, self).__init__()
        self.conv1 = nn.Conv2d(1, 32, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, padding=1)
        self.pool = nn.MaxPool2d(2, 2)
        self.fc1 = nn.Linear(64 * 16 * 16, 128)
        self.fc2 = nn.Linear(128, 4)

    def forward(self, x):
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        x = x.view(-1, 64 * 16 * 16)
        x = F.relu(self.fc1(x))
        x = self.fc2(x)
        return x
```

## How to Run the Project

### 1. Prepare the Data
Organize the dataset in the `data/` directory:

```
player-position/
├── data/
│   ├── train/
│   │     ├── Forward/
│   │     ├── Midfielder/
│   │     ├── Defender/
│   │     └── Goalkeeper/
│   └── test/
```

Ensure each subfolder in `train/` contains the corresponding player position images.

### 2. Train the Model
Run the following command to train the model:

```bash
python player_position_pytorch.py
```

The model will be trained for 10 epochs and the trained weights will be saved as `player_position_model.pth`.

### 3. Make Predictions on Test Data
After training, the script automatically generates predictions for images in the `data/test/` directory and saves them to `submission_pytorch.csv`.

## Output
The model outputs a `submission_pytorch.csv` file with predicted player positions:

Example:
```
Position
Forward
Midfielder
Defender
Goalkeeper
```

## Customization

### Adjust Hyperparameters
Modify these lines in `player_position_pytorch.py` to adjust learning rate, batch size, and number of epochs:

```python
train_loader = DataLoader(train_dataset, batch_size=16, shuffle=True)
optimizer = optim.Adam(model.parameters(), lr=0.001)
for epoch in range(10):
```

### Evaluate Model Performance
Consider adding metrics like accuracy calculation on the validation set.

## Troubleshooting

1. **Shape mismatch error**: Ensure images are loaded in grayscale and resized correctly to `(64, 64)`.
2. **Missing data**: Verify the training and test directories are populated with images.
3. **GPU usage**: If you want to use GPU acceleration, move the model and tensors to CUDA:

```python
model = model.to('cuda')
images = images.to('cuda')
```

## Future Improvements
- Data Augmentation (e.g., random flips, rotations) for better generalization.
- Increase model complexity for better accuracy.
- Implement performance evaluation (e.g., confusion matrix, precision/recall).

# Keras

This project uses a Convolutional Neural Network (CNN) implemented with Keras to classify soccer player positions (Forward, Midfielder, Defender, Goalkeeper) based on heatmap images.

## Project Structure

```
player-position/
├── data/
│   ├── train/
│   │    ├── Forward/
│   │    ├── Midfielder/
│   │    ├── Defender/
│   │    └── Goalkeeper/
│   └── test/
├── player_position_keras.py
├── player_position_model.h5
└── submission_keras.csv
```

- **data/train/**: Contains training images organized by player position.
- **data/test/**: Contains test images for prediction.
- **player_position_keras.py**: Main script for training and predicting.
- **player_position_model.h5**: Saved Keras model after training.

## Requirements

Ensure you have the following libraries installed:

```bash
pip install tensorflow numpy pandas opencv-python scikit-learn
```

## Model Architecture

The CNN model consists of:

- **Conv2D (32 filters, kernel size 3x3, ReLU activation)**
- **MaxPooling2D (pool size 2x2)**
- **Conv2D (64 filters, kernel size 3x3, ReLU activation)**
- **MaxPooling2D (pool size 2x2)**
- **Flatten Layer**
- **Dense Layer (128 units, ReLU activation)**
- **Output Layer (4 units, softmax activation)**

## How the Code Works

### 1. Data Preparation

The `prepare_dataset()` function:
- Loads grayscale images from the `data/train` directory.
- Resizes images to 64x64 pixels.
- Normalizes pixel values to [0, 1].
- Labels each image according to its folder.

### 2. Model Building

The `build_model()` function:
- Constructs a CNN with two convolutional layers followed by max-pooling.
- Flattens the output and passes it through a fully connected layer.
- Uses the Adam optimizer and sparse categorical crossentropy loss.

### 3. Model Training

The `train_model()` function:
- Splits the dataset into training (80%) and validation (20%) sets.
- Trains the model for 10 epochs with a batch size of 16.
- Saves the trained model as `player_position_model.h5`.

### 4. Prediction on Test Images

The `predict_test_images()` function:
- Loads and preprocesses images from the `data/test` directory.
- Predicts the player position for each image.
- Saves predictions to `submission_keras.csv`.

## Usage

1. Ensure the dataset is organized as shown in the directory structure.

2. Run the script to train the model and generate predictions:

```bash
python player_position_keras.py
```

3. The model is saved as `player_position_model.h5` and predictions are stored in `submission_keras.csv`.

## Example Output

The `submission_keras.csv` will contain:

```
Position
Forward
Midfielder
Defender
Goalkeeper
```

## Customization

- **Change Model Architecture**: Modify the `build_model()` function to experiment with deeper networks.
- **Adjust Hyperparameters**: Modify learning rate, batch size, or epochs in `train_model()`.
- **Add More Classes**: Update the `categories` dictionary in `prepare_dataset()`.

## Troubleshooting

1. **Image Loading Issues**:
   Ensure the images are in the correct folders and are valid.

2. **Prediction Errors**:
   Confirm that the test images are preprocessed similarly to the training images.

3. **Low Accuracy**:
   Increase the number of epochs, add data augmentation, or tune the model.

# TensorFlow

This project is a TensorFlow-based implementation for classifying soccer player positions from heatmap images. The model categorizes players into one of four roles: Forward, Midfielder, Defender, and Goalkeeper. It uses a convolutional neural network (CNN) built with pure TensorFlow, including manual layer definitions and custom training loops.

## Project Structure

```
player-position/
├── data/
│   ├── train/
│   │    ├── Forward/
│   │    ├── Midfielder/
│   │    ├── Defender/
│   │    └── Goalkeeper/
│   └── test/
├── player_position_tensorflow.py
├── player_position_model_tf/
└── submission_tf.csv
```

- `data/train/`: Directory containing training images categorized into four folders: Forward, Midfielder, Defender, Goalkeeper.
- `data/test/`: Directory containing test images for prediction.
- `player_position_tensorflow.py`: Main script for training and evaluating the model.
- `player_position_model_tf/`: Directory containing the model after being saved.
- `submission_tf.csv`: Output file containing predicted player positions.

## Prerequisites

Ensure you have the following packages installed:

```bash
pip install tensorflow numpy pandas opencv-python scikit-learn
```

## Model Overview

The implemented CNN model architecture includes:

- **Conv Layer 1**: 32 filters of size (3x3), ReLU activation
- **Max Pooling**: (2x2)
- **Conv Layer 2**: 64 filters of size (3x3), ReLU activation
- **Max Pooling**: (2x2)
- **Fully Connected Layer**: 128 units, ReLU activation
- **Output Layer**: 4 units (for 4 player positions), Softmax activation

## How to Run the Project

### 1. Organize Data

Ensure your `data/train/` directory is structured like this:

```
data/train/
├── Forward/
├── Midfielder/
├── Defender/
└── Goalkeeper/
```

Place test images in `data/test/`.

### 2. Train the Model

Run the script to train the model:

```bash
python player_position_tensorflow.py
```

The model will:
- Load and preprocess the training dataset (64x64 grayscale images).
- Train for 10 epochs using Adam optimizer and cross-entropy loss.
- Save the trained model to `player_position_model_tf`.

### 3. Predict Test Images

The script automatically predicts the player positions for test images and saves the results in `submission_tf.csv`.

### Example Output

The output CSV (`submission_tf.csv`) will have the following structure:

```
Position
Forward
Midfielder
Defender
Goalkeeper
...
```

## Customization

### 1. Modify Model Hyperparameters

Adjust learning rate, batch size, and epochs by changing the following lines in the `train_model()` function:

```python
learning_rate = 0.001
epochs = 10
batch_size = 16
```

### 2. Add More Classes

Update the `categories` dictionary in `prepare_dataset()` to add more categories.

### 3. Change Image Size

Modify the image size in both `load_images_from_folder()` and model input shape:

```python
img = cv2.resize(img, (NEW_WIDTH, NEW_HEIGHT))
input_shape=(NEW_WIDTH, NEW_HEIGHT, 1)
```

## Performance Considerations

1. Ensure training and test images are of consistent size (64x64 by default).
2. Consider increasing the dataset size for better model generalization.

## Troubleshooting

- **Tensor shape mismatch**: Ensure images are correctly resized and reshaped.
- **Low accuracy**: Adjust learning rate, increase model complexity, or augment the dataset.

## Acknowledgments

This project leverages TensorFlow's low-level API for a fully customizable deep learning pipeline.

## Future Improvements

1. Implement data augmentation for improved generalization.
2. Explore advanced architectures (e.g., ResNet or EfficientNet).
3. Deploy the trained model using TensorFlow Serving for real-time inference.

## License
Feel free to use and modify the code for your projects (MIT License).

## Contact
For questions or contributions, feel free to reach out!

