import torch
from torch.utils.data import Dataset, DataLoader
import numpy as np
import torchvision.transforms as transforms

class ImageDataset(Dataset):
    def __init__(self, file_path_X, file_path_y, transform=None):
        self.data = np.fromfile(file_path_X, dtype=np.float32).reshape(-1, 64, 64)
        self.labels = np.fromfile(file_path_y, dtype=np.int32)
        self.transform = transform

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        image = self.data[idx]
        label = self.labels[idx]
        if self.transform:
            image = self.transform(image)
        image = torch.tensor(image, dtype=torch.float32).unsqueeze(0).clone().detach()  # Fix warning
        return image, torch.tensor(label, dtype=torch.long)

def load_data(file_path_X, file_path_y, batch_size=100):
    transform = transforms.Compose([
        transforms.ToPILImage(),
        transforms.RandomHorizontalFlip(),
        transforms.RandomRotation(10),
        transforms.ToTensor()
    ])
    dataset = ImageDataset(file_path_X, file_path_y, transform=transform)
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True, num_workers=4)
    return dataloader
