import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

class CNN(nn.Module):
    def __init__(self):
        super(CNN, self).__init__()
        self.conv1 = nn.Conv2d(1, 8, kernel_size=3, stride=1, padding=1)
        self.bn1 = nn.BatchNorm2d(8)
        self.conv2 = nn.Conv2d(8, 16, kernel_size=3, stride=1, padding=1)
        self.bn2 = nn.BatchNorm2d(16)
        self.fc1 = nn.Linear(16 * 16 * 16, 64)
        self.fc2 = nn.Linear(64, 4)

    def forward(self, x):
        x = F.relu(self.bn1(self.conv1(x)))
        x = F.max_pool2d(x, 2)
        x = F.relu(self.bn2(self.conv2(x)))
        x = F.max_pool2d(x, 2)
        x = x.view(x.size(0), -1)
        x = F.relu(self.fc1(x))
        x = self.fc2(x)
        return x

def save_weights_binary(model, file_path):
    with open(file_path, 'wb') as f:
        for k, v in model.state_dict().items():
            print(f"Saving {k}: shape {v.shape}, dtype {v.dtype}")
            np.save(f, v.cpu().numpy())

def load_weights_binary(model, file_path):
    state_dict = model.state_dict()
    with open(file_path, 'rb') as f:
        for k in state_dict.keys():
            state_dict[k] = torch.from_numpy(np.load(f))
            print(f"Loading {k}: shape {state_dict[k].shape}, dtype {state_dict[k].dtype}")
    model.load_state_dict(state_dict)
