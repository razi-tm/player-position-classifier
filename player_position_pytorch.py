import os
import cv2
import torch
import numpy as np
import pandas as pd
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from sklearn.model_selection import train_test_split

# Define dataset class
class HeatmapDataset(Dataset):
    def __init__(self, image_paths, labels, transform=None):
        self.image_paths = image_paths
        self.labels = labels
        self.transform = transform
    
    def __len__(self):
        return len(self.image_paths)
    
    def __getitem__(self, idx):
        img_path = self.image_paths[idx]
        # Read the image as grayscale; shape will be (64, 64) after resize
        image = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
        image = cv2.resize(image, (64, 64))
        
        label = self.labels[idx]
        
        if self.transform:
            # Let the transform handle conversion (e.g., ToTensor will convert (64,64) to (1,64,64))
            image = self.transform(image)
        else:
            # If no transform, add channel dimension manually
            image = np.expand_dims(image, axis=0).astype(np.float32)
            image = torch.tensor(image, dtype=torch.float32)
        
        return image, torch.tensor(label, dtype=torch.long)



# CNN Model
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

# Load images
def load_dataset():
    data_dir = "data/train"
    categories = {"Forward": 0, "Midfielder": 1, "Defender": 2, "Goalkeeper": 3}
    image_paths, labels = [], []
    
    for category, label in categories.items():
        folder = os.path.join(data_dir, category)
        for filename in os.listdir(folder):
            image_paths.append(os.path.join(folder, filename))
            labels.append(label)
    
    return image_paths, labels, categories

# Train the model
def train_model():
    image_paths, labels, categories = load_dataset()
    X_train, X_val, y_train, y_val = train_test_split(image_paths, labels, test_size=0.2, random_state=42)
    
    transform = transforms.Compose([transforms.ToTensor()])
    train_dataset = HeatmapDataset(X_train, y_train, transform=transform)
    val_dataset = HeatmapDataset(X_val, y_val, transform=transform)
    
    train_loader = DataLoader(train_dataset, batch_size=16, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=16, shuffle=False)
    
    model = PlayerPositionCNN()
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)
    
    for epoch in range(10):
        model.train()
        for images, labels in train_loader:
            print("Input shape: ", images.shape)  # Should be [batch_size, 1, 64, 64]
            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
        
        print(f"Epoch {epoch+1}/10, Loss: {loss.item():.4f}")
    
    torch.save(model.state_dict(), "player_position_model.pth")
    return model, categories

# Predict test images
def predict_test_images(model, categories):
    test_dir = "data/test"
    test_images = []
    test_filenames = []
    
    for filename in sorted(os.listdir(test_dir), key=lambda x: int(x.split('.')[0])):
        img_path = os.path.join(test_dir, filename)
        image = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
        image = cv2.resize(image, (64, 64))
        image = np.expand_dims(image, axis=0) / 255.0
        test_images.append(image)
        test_filenames.append(filename)
    
    test_images = torch.tensor(np.array(test_images), dtype=torch.float32)
    
    model.eval()
    predictions = model(test_images).detach().numpy()
    predicted_labels = [list(categories.keys())[np.argmax(pred)] for pred in predictions]
    
    df = pd.DataFrame(predicted_labels, columns=["Position"])
    df.to_csv("submission_pytorch.csv", index=False)
    print("Predictions saved to submission.csv")

if __name__ == "__main__":
    model, categories = train_model()
    predict_test_images(model, categories)
