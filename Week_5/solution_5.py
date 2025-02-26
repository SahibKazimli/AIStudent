import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt

# Task 1: Linear Regression and Autograd

# Generate data
np.random.seed(0)
x = np.random.rand(100, 1)
y = 2 + 3 * x + np.random.rand(100, 1)

# Initialize parameters
w = np.random.rand(1, 1)
b = np.random.rand(1)
learning_rate = 0.1
num_epochs = 1000

# Gradient Descent Implementation
def forward(x, w, b):
    return np.dot(x, w) + b

def compute_loss(y, y_hat):
    return np.mean((y - y_hat) ** 2)

def backward(x, y, y_hat, w, b):
    m = x.shape[0]
    dw = -2/m * np.dot(x.T, (y - y_hat))
    db = -2/m * np.sum(y - y_hat)
    return dw, db

for epoch in range(num_epochs):
    y_hat = forward(x, w, b)
    loss = compute_loss(y, y_hat)
    dw, db = backward(x, y, y_hat, w, b)
    w -= learning_rate * dw
    b -= learning_rate * db
    if epoch % 100 == 0:
        print(f"Epoch {epoch}: Loss = {loss:.4f}")

# Task 2: MNIST and Multilayer Perceptron

# Generate toy data
np.random.seed(42)
x = np.random.rand(100, 1)
y = (x > 0.5).astype(float)

# Initialize parameters
input_dim, hidden_dim, output_dim = 1, 5, 1
W1, b1 = np.random.randn(input_dim, hidden_dim) * 0.1, np.zeros((1, hidden_dim))
W2, b2 = np.random.randn(hidden_dim, output_dim) * 0.1, np.zeros((1, output_dim))

def sigmoid(z):
    return 1 / (1 + np.exp(-z))

def forward(x, W1, b1, W2, b2):
    z1 = np.dot(x, W1) + b1
    a1 = sigmoid(z1)
    z2 = np.dot(a1, W2) + b2
    a2 = sigmoid(z2)
    return z1, a1, z2, a2

def backward(x, y, z1, a1, z2, a2, W1, W2):
    m = x.shape[0]
    dz2 = a2 - y
    dW2, db2 = np.dot(a1.T, dz2) / m, np.sum(dz2, axis=0, keepdims=True) / m
    da1 = np.dot(dz2, W2.T)
    dz1 = da1 * (a1 * (1 - a1))
    dW1, db1 = np.dot(x.T, dz1) / m, np.sum(dz1, axis=0, keepdims=True) / m
    return dW1, db1, dW2, db2

def update_parameters(W1, b1, W2, b2, dW1, db1, dW2, db2, learning_rate):
    W1 -= learning_rate * dW1
    b1 -= learning_rate * db1
    W2 -= learning_rate * dW2
    b2 -= learning_rate * db2
    return W1, b1, W2, b2

for epoch in range(1000):
    z1, a1, z2, a2 = forward(x, W1, b1, W2, b2)
    loss = np.mean(-y * np.log(a2) - (1 - y) * np.log(1 - a2))
    dW1, db1, dW2, db2 = backward(x, y, z1, a1, z2, a2, W1, W2)
    W1, b1, W2, b2 = update_parameters(W1, b1, W2, b2, dW1, db1, dW2, db2, 0.1)
    if epoch % 100 == 0:
        print(f"Epoch {epoch}: Loss = {loss:.4f}")

# Task 3: MNIST and Convolutional Neural Networks

# Data Preparation
transform = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.5,), (0.5,))])
train_dataset = datasets.MNIST(root='./data', train=True, transform=transform, download=True)
test_dataset = datasets.MNIST(root='./data', train=False, transform=transform, download=True)
train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=64, shuffle=False)

# Define CNN Model
class CNN(nn.Module):
    def __init__(self):
        super(CNN, self).__init__()
        self.conv1 = nn.Conv2d(1, 16, kernel_size=3, stride=1, padding=1)
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2, padding=0)
        self.conv2 = nn.Conv2d(16, 32, kernel_size=3, stride=1, padding=1)
        self.fc1 = nn.Linear(32 * 7 * 7, 128)
        self.fc2 = nn.Linear(128, 10)
    
    def forward(self, x):
        x = self.pool(torch.relu(self.conv1(x)))
        x = self.pool(torch.relu(self.conv2(x)))
        x = x.view(-1, 32 * 7 * 7)
        x = torch.relu(self.fc1(x))
        x = self.fc2(x)
        return x

# Initialize Model, Loss, and Optimizer
model = CNN()
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)

# Training Loop
num_epochs = 5
for epoch in range(num_epochs):
    model.train()
    running_loss = 0.0
    for images, labels in train_loader:
        optimizer.zero_grad()
        outputs = model(images)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
        running_loss += loss.item()
    print(f"Epoch {epoch+1}/{num_epochs}, Loss: {running_loss/len(train_loader):.4f}")

# Evaluation
model.eval()
correct, total = 0, 0
with torch.no_grad():
    for images, labels in test_loader:
        outputs = model(images)
        _, predicted = torch.max(outputs, 1)
        total += labels.size(0)
        correct += (predicted == labels).sum().item()

print(f"Accuracy: {100 * correct / total:.2f}%")
