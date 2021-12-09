import os
import cv2
import random
import pickle

import pandas as pd
import numpy as np

import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.models as models
from torchvision import transforms
from torch.utils.data import Dataset, DataLoader

import matplotlib.pyplot as plt
from sklearn.metrics import accuracy_score, confusion_matrix
from sklearn.model_selection import train_test_split

from google.colab import drive
drive.mount('/content/gdrive')


#CONFIG

cfg = {
    'images_path': '/content/gdrive/MyDrive/Pisici/Images',
    'train_path': '/content/gdrive/MyDrive/Pisici/Dataset/train_dataset.pickle',
    'test_path': '/content/gdrive/MyDrive/Pisici/Dataset/test_dataset.pickle',
    'img_size': 224,
    'learning_rate': 0.001,
}

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


#CREATING THE DATASET

def preprocess_image(path, cfg):
    preprocess = transforms.Compose([
        transforms.ToTensor(),
        # Thats the normalization that pytorch models accept
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])

    # Reshaping and normalizing the images so the vgg-16 model accepts them.
    img = cv2.resize(cv2.imread(path, cv2.IMREAD_COLOR), (cfg['img_size'], cfg['img_size'])) # img shape -> 3 x 224 x 224
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

    img = preprocess(img)
    
    return img

dataset = []

# Useful dictionaries for encoding the labels
label_to_id = dict()
id_to_label = dict()

# Iterate through the folders
for idx, emotion_folder_name in enumerate(os.listdir(cfg['images_path'])):

    # Encode the labels
    label_to_id[emotion_folder_name] = idx
    id_to_label[idx] = emotion_folder_name

    emotion_folder_path = os.path.join(cfg['images_path'], emotion_folder_name)

    # Iterate through the images in each folder
    for image_name in os.listdir(emotion_folder_path):
        image_path = os.path.join(emotion_folder_path, image_name)
        dataset.append([preprocess_image(image_path, cfg), idx])

# Split the dataset into train and test
images = [x[0] for x in dataset]
labels = [x[1] for x in dataset]

X_train, X_test, y_train, y_test = train_test_split(images, labels, test_size=0.1, random_state=7, shuffle=True)


train_data = list(zip(X_train, y_train))
test_data = list(zip(X_test, y_test))

# Saving the dataset as pickled form
with open(cfg['train_path'], 'wb') as f:
    pickle.dump(train_data, f)

with open(cfg['test_path'], 'wb') as f:
    pickle.dump(test_data, f)




#DATASET CLASS

class CatsImageDataset(Dataset):
    def __init__(self, dataset_file_path):
        with open(dataset_file_path, 'rb') as f:
            self.dataset = pickle.load(f)

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx):
        # Return image, label
        return dataset[idx][0], dataset[idx][1]



#MODEL CLASS

class Net(nn.Module):
   
    def __init__(self):
        super().__init__()
        self.vgg16 = models.vgg16(pretrained=True)

        # Replace last layer last layer
        self.vgg16.classifier[6] = nn.Linear(4096, 5)

    def forward(self, x):
        x = self.vgg16(x)

        # No need for softmax as we are using CrossEntropy as the loss function
        return x

# Function that saves the model in the google drive
def save_model(model, cfg, model_save_name):
    ''' model_save_name should be a string ending in .pt
    '''
    path = f'/content/gdrive/MyDrive/Pisici/Saved-Models/{model_save_name}'
    torch.save(model.state_dict(), path)

# Initiating our model
model = Net()
model.to(device)

# GD optimizer and out loss function
optimizer = torch.optim.SGD(model.parameters(), cfg['learning_rate'])

# TODO: SOFTMAX
loss = nn.CrossEntropyLoss()

# Initialize train and test dataloaders
training_dataset = CatsImageDataset(cfg['train_path'])
test_dataset = CatsImageDataset(cfg['test_path'])

train_dataloader = DataLoader(training_dataset, batch_size=8, shuffle=False)
test_dataloader = DataLoader(test_dataset, batch_size=1, shuffle=False)



#TRAINING ROUTINE

def train(model, train_dataloader, loss, epochs):
    X_train = iter(train_dataloader)
    idx = 0

    while True:
        try:
            images, labels = next(X_train)
            images, labels = images.to(device), labels.to(device)
            y_preds = model(images)
        except StopIteration:
            break

        # Reset the gradient
        model.zero_grad()

        # Compute the loss
        output_loss = loss(y_preds, labels)

        if idx % 10 == 0:
            print(f'Loss: {output_loss}')

        # Sends the back propagation signal
        output_loss.backward()

        optimizer.step()

        idx += 1



#TESTING ROUTINE

def test(model, test_dataloader):
    X_test = iter(test_dataloader)
    labels = torch.tensor([])
    predictions = torch.tensor([])

    labels, predictions = labels.to(device), predictions.to(device)

    while True:
        try:
            image, label = next(X_test)
            image, label = image.to(device), label.to(device)

            labels = torch.cat((labels, label))

            with torch.no_grad():
                y_pred = model(image)
        except StopIteration:
            break

        # Get the predictions
        logSoftMax = nn.LogSoftmax()
        y_pred = logSoftMax(y_pred)
        y_pred = torch.argmax(y_pred, dim=-1)

        # Save the predictions
        predictions = torch.cat((predictions, y_pred))
    
    # Get the tensors to host memory so .numpy() works
    labels, predictions = labels.cpu(), predictions.cpu()

    print(labels)
    print(predictions)

    accuracy = accuracy_score(labels.numpy(), predictions.numpy())
    conf_mat = confusion_matrix(labels.numpy(), predictions.numpy())
    print(f'Accuracy: {accuracy}')
    print(f'Confusion matrix: {conf_mat}')

# Train
train(model, train_dataloader, loss, 1)

# Test
test(model, test_dataloader)