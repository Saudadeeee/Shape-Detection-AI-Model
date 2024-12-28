import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import csv
import sys

csv.field_size_limit(2**31 - 1)  

class CNN(nn.Module):
    def __init__(self):
        super(CNN, self).__init__()
        self.conv1 = nn.Conv2d(1, 8, kernel_size=3, stride=1, padding=1)
        self.bn1 = nn.BatchNorm2d(8)  # Add batch normalization
        self.conv2 = nn.Conv2d(8, 16, kernel_size=3, stride=1, padding=1)
        self.bn2 = nn.BatchNorm2d(16)  # Add batch normalization
        self.fc1 = nn.Linear(16 * 16 * 16, 64)
        self.fc2 = nn.Linear(64, 4)

    def forward(self, x):
        x = F.relu(self.bn1(self.conv1(x)))  # Apply batch normalization
        x = F.max_pool2d(x, 2)
        x = F.relu(self.bn2(self.conv2(x)))  # Apply batch normalization
        x = F.max_pool2d(x, 2)
        x = x.view(x.size(0), -1)
        x = F.relu(self.fc1(x))
        x = self.fc2(x)
        return x

def save_weights(model, file_path):
    weights = {k: v.cpu().numpy() for k, v in model.state_dict().items()}
    with open(file_path, 'w', newline='') as f:
        writer = csv.writer(f)
        for k, v in weights.items():
            writer.writerow([k, v.tolist()])

def load_weights(model, file_path):
    state_dict = model.state_dict()
    with open(file_path, 'r') as f:
        reader = csv.reader(f)
        for row in reader:
            k = row[0]
            v = np.array(eval(row[1]), dtype=np.float32)
            state_dict[k] = torch.from_numpy(v)
    model.load_state_dict(state_dict)
