# -*- coding: utf-8 -*-

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

file_path = r"C:\Users\miko9\Downloads\datasets\\"
train = pd.read_csv(file_path + 'train32.csv')
test = pd.read_csv(file_path + 'test32.csv')
print(train)
print(test)

# 0: good textile, 1: damaged textile
# Train Data
train['index'] = train['index'] - 48000
train_good = train[train['indication_value'] == 0]
train_damaged = train[train['indication_value'] != 0]
train_damaged['indication_value'] = 1

# Test Data
test['index'] = test['index']
test_good = test[test['indication_value'] == 0]
test_damaged = test[test['indication_value'] != 0]
test_damaged['indication_value'] = 1

print(test_good.count())
print(test_damaged[test['angle'] == 120].count())
print(test_damaged[test['angle'] == 20].count())

train_table = pd.concat([train_good, train_damaged[train['angle'] == 20], train_damaged[train['angle'] == 120]])
test_table = pd.concat([test_good, test_damaged[test['angle'] == 20], test_damaged[test['angle'] == 120]])

import h5py
import keras

f = h5py.File(file_path + 'train32.h5', 'r')
##a_group_key = list(f.keys())[0]
data = list(f['images'])

x_train = []
y_train = []
idx = train_table['index'].astype('int')
indication_value = train_table['indication_value'].astype('int')
for i in idx:
    x_train.append(data[i])
    y_train.append(indication_value.loc[idx[i]])

x_train = np.array(x_train)
y_train = np.array(y_train)
y_train = keras.utils.to_categorical(y_train, num_classes=2)

print(x_train.shape)
print(y_train.shape)

f = h5py.File(file_path + 'test32.h5', 'r')
#a_group_key = list(f.keys())[0]
data = list(f['images'])

x_test = []
y_test = []
idx = test_table['index'].astype('int')
indication_value = test_table['indication_value'].astype('int')
for i in idx:
    x_test.append(data[i])
    y_test.append(indication_value.loc[idx[i]])

x_test = np.array(x_test)
y_test = np.array(y_test)
y_test = keras.utils.to_categorical(y_test, num_classes=2)

print(x_test.shape)
print(y_test.shape)

# Build model
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
import torchvision.transforms as transforms
import torch.optim as optim
from torchvision.models import resnet18

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

# Build ResNet50 model
model = resnet18(pretrained = True)
model.conv1 = nn.Conv2d(1, 64, kernel_size=(7, 7), stride=(2, 2), padding=(3, 3),bias=False)

# Connect to fully connnected layer
model.fc = nn.Sequential(
    nn.Linear(512,512),
    nn.ReLU(inplace=True),
    nn.Linear(512,64),
    nn.ReLU(inplace=True),
    nn.Linear(64,2)
)

def init_weights(m):
    if isinstance(m, nn.Conv2d) or isinstance(m, nn.Linear):
        nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
        if m.bias is not None:
            nn.init.constant_(m.bias, 0)

model.apply(init_weights)
model.to(device)

#####
from sklearn.model_selection import train_test_split

# Set data loader
class cifar10Dataset(Dataset):
    def __init__(self, imgs, labels, transform=None):
        self.imgs = imgs
        self.labels = labels
        self.transforms = transform

    def __len__(self):
        return len(self.imgs)

    def __getitem__(self, index):
        x = self.imgs[index]
        y = self.labels[index]

        if self.transforms:
            x = self.transforms(x)

        x = x.float()
        return x, y

# Split train and validation
x_train, x_val, y_train, y_val = train_test_split(x_train,
                                                  y_train,
                                                  test_size=0.2,
                                                  random_state=100,
                                                  shuffle=True)

# Add subset logic here
#subset_size = 500  # Define the size of your subset
#x_train = x_train[:subset_size]
#y_train = y_train[:subset_size]
#x_val = x_val[:subset_size]
#y_val = y_val[:subset_size]
#x_test = x_test[:subset_size]
#y_test = y_test[:subset_size]

learningRate = 0.001
batch_size = 128 ## change it later to 64

# Set loss function and optimiser
criterion = nn.CrossEntropyLoss()
#optimizer = optim.SGD(model.parameters(),lr = learningRate)
optimizer = optim.Adam(model.parameters(),lr = learningRate)

# stats = ((0.5), (0.5))
transform = transforms.Compose([transforms.ToTensor(),
                                # transforms.RandomCrop(32, padding=4, padding_mode='reflect'),
                                # transforms.RandomHorizontalFlip(),
                                # transforms.Normalize(*stats,inplace=True)
                               ])

train_dataset = cifar10Dataset(x_train, y_train, transform)
val_dataset = cifar10Dataset(x_val, y_val, transform)
test_dataset = cifar10Dataset(x_test, y_test, transform)

#COMMENT THESE LINES LATER
#train_loader = DataLoader(dataset=train_dataset, batch_size=batch_size, shuffle=True)
#val_loader = DataLoader(dataset=val_dataset, batch_size=batch_size, shuffle=True)
#test_loader = DataLoader(dataset=test_dataset, batch_size=batch_size, shuffle=True)
#UNCOMMENT THIS LATER##
train_loader = DataLoader(dataset=train_dataset, batch_size= batch_size, shuffle=True)
val_loader = DataLoader(dataset=val_dataset, batch_size= batch_size, shuffle=True)
test_loader = DataLoader(dataset=test_dataset, batch_size= batch_size, shuffle=True)

####################################################### Optimized Training Loop ###############################################
# Add the scheduler
scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=5, verbose=True)

num_epoch = 100
best_val_accuracy = 0.0

# Initialize lists to store metrics
train_losses = []
val_losses = []
train_accuracies = []
val_accuracies = []

for epoch in range(num_epoch):
    model.train()
    running_loss = 0.0
    correct_train = 0
    total_train = 0

    for images, labels in train_loader:
        images, labels = images.to(device), labels.to(device)
        optimizer.zero_grad()
        outputs = model(images)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        # Accumulate training loss and accuracy
        running_loss += loss.item()
        _, predicted = torch.max(outputs.data, 1)
        _, labels = torch.max(labels, 1)
        total_train += labels.size(0)
        correct_train += (predicted == labels).sum().item()

    train_loss = running_loss / len(train_loader)
    train_accuracy = correct_train / total_train
    train_losses.append(train_loss)
    train_accuracies.append(train_accuracy)

    model.eval()
    running_val_loss = 0.0
    correct = 0
    total = 0

    with torch.no_grad():
        for images, labels in val_loader:
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            loss = criterion(outputs, labels)  # Calculate validation loss
            running_val_loss += loss.item()
            _, predicted = torch.max(outputs.data, 1)
            _, labels = torch.max(labels, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

        val_loss = running_val_loss / len(val_loader)
        val_losses.append(val_loss)  # Store validation loss
        val_accuracy = correct / total
        val_accuracies.append(val_accuracy)  # Store validation accuracy

        # Adjust the learning rate based on validation loss
        scheduler.step(val_loss)

        if val_accuracy > best_val_accuracy:
            best_val_accuracy = val_accuracy
            torch.save(model.state_dict(), file_path + 'best_model.pth')
            print("Saved Best Model!")

        print(f'Epoch [{epoch+1}/{num_epoch}], Train Loss: {train_loss:.4f}, Train Accuracy: {train_accuracy:.4f}, Val Loss: {val_loss:.4f}, Validation Accuracy: {val_accuracy:.4f}, BEST Accuracy: {best_val_accuracy:.4f}')

# Save metrics after training is complete
torch.save({
    'train_losses': train_losses,
    'val_losses': val_losses,
    'train_accuracies': train_accuracies,
    'val_accuracies': val_accuracies,
}, file_path + 'metrics.pth')

print('Training Finished and Metrics Saved!')


####################################################### Optimized Training Loop ###############################################
model.load_state_dict(torch.load(file_path + 'best_model.pth'))
model.eval()

total = 0
correct = 0

for images, labels in test_loader:
    images, labels = images.to(device), labels.to(device)
    batch_outputs = model(images)
    _, predicted = torch.max(batch_outputs, 1)
    _, labels = torch.max(labels, 1)
    total += labels.size(0)
    correct += (predicted == labels).sum().item()

test_accuracy = correct / total
print('Test accuracy: ', test_accuracy)

##Cheeno added this
import torch
import torchvision.transforms as transforms
from PIL import Image
import numpy as np
import os

# Function to preprocess a single image
def preprocess_image(image_path):
    image = Image.open(image_path).convert('L')  # Open and convert to grayscale
    transform = transforms.Compose([
        transforms.Resize((32, 32)),  # Resize to 32x32
        transforms.ToTensor(),
        # transforms.Normalize(*stats,inplace=True) # Use the same normalization as training if applicable
    ])
    image = transform(image)
    image = image.unsqueeze(0) # Add batch dimension
    return image

# Directory containing the images
#image_directory = '/content/Images'
image_directory = r"C:\Users\miko9\Downloads\datasets\Images"

# Loop through the images in the directory
for filename in os.listdir(image_directory):
    if filename.endswith(('.png', '.jpg', '.jpeg')):  # Check for image file extensions
        image_path = os.path.join(image_directory, filename)
        try:
            input_image = preprocess_image(image_path)
            input_image = input_image.to(device)

            with torch.no_grad():
                outputs = model(input_image)
                _, predicted = torch.max(outputs, 1)

            print(f"Image: {filename}, Predicted class: {predicted.item()}")
        except Exception as e:
            print(f"Error processing image {filename}: {e}")

#Plotting using SGD Optimizer

#import matplotlib.pyplot as plt

# Plot training vs validation loss
#plt.figure(figsize=(10, 5))
#plt.plot(train_losses, label="Train Loss (SGD)")
#plt.plot(val_losses, label="Validation Loss (SGD)")
#plt.xlabel("Epochs")
#plt.ylabel("Loss")
#plt.title("Loss Curve")
#plt.legend()
#plt.show()

# Plot training vs validation accuracy
#plt.figure(figsize=(10, 5))
#plt.plot(train_accuracies, label="Train Accuracy (SGD)")
#plt.plot(val_accuracies, label="Validation Accuracy (SGD)")
#plt.xlabel("Epochs")
#plt.ylabel("Accuracy")
#plt.title("Accuracy Curve")
#plt.legend()
#plt.show()

#Plotting using Adam Optimizer

import matplotlib.pyplot as plt

# Plot training vs validation loss
plt.figure(figsize=(10, 5))
plt.plot(train_losses, label="Train Loss (Adam)")
plt.plot(val_losses, label="Validation Loss (Adam)")
plt.xlabel("Epochs")
plt.ylabel("Loss")
plt.title("Loss Curve")
plt.legend()
plt.show()

# Plot training vs validation accuracy
plt.figure(figsize=(10, 5))
plt.plot(train_accuracies, label="Train Accuracy (Adam)")
plt.plot(val_accuracies, label="Validation Accuracy (Adam)")
plt.xlabel("Epochs")
plt.ylabel("Accuracy")
plt.title("Accuracy Curve")
plt.legend()
plt.show()