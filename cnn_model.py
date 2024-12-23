import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

class CNN(nn.Module):
    def __init__(self):
        super(CNN, self).__init__()
        self.conv1 = nn.Conv2d(1, 32, kernel_size=3, stride=1, padding=1)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, stride=1, padding=1)
        self.conv3 = nn.Conv2d(64, 128, kernel_size=3, stride=1, padding=1)
        self.fc1 = nn.Linear(128 * 8 * 8, 256)
        self.fc2 = nn.Linear(256, 128)
        self.fc3 = nn.Linear(128, 4)

    def forward(self, x):
        x = F.relu(self.conv1(x))
        x = F.max_pool2d(x, 2)
        x = F.relu(self.conv2(x))
        x = F.max_pool2d(x, 2)
        x = F.relu(self.conv3(x))
        x = F.max_pool2d(x, 2)
        x = x.view(x.size(0), -1)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return x

def save_weights(model, file_path):
    weights = {k: v.cpu().numpy() for k, v in model.state_dict().items()}
    with open(file_path, 'wb') as f:
        for k, v in weights.items():
            f.write(v.tobytes())

def load_weights(model, file_path):
    state_dict = model.state_dict()
    with open(file_path, 'rb') as f:
        for k in state_dict.keys():
            size = np.prod(state_dict[k].shape)
            state_dict[k] = torch.from_numpy(np.frombuffer(f.read(size * 4), dtype=np.float32).reshape(state_dict[k].shape))
    model.load_state_dict(state_dict)
