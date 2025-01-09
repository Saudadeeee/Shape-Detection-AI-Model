import torch
from torch.utils.data import Dataset, DataLoader
import numpy as np
import torchvision.transforms as transforms
from sklearn.utils import resample

class ImageDataset(Dataset):
    def __init__(self, file_path_X, file_path_y, transform=None):
        self.data = np.fromfile(file_path_X, dtype=np.float32).reshape(-1, 64, 64)
        self.labels = np.fromfile(file_path_y, dtype=np.int32)
        self.transform = transform

        # Balance the dataset
        self.data, self.labels = self.balance_dataset(self.data, self.labels)

    def balance_dataset(self, data, labels):
        unique, counts = np.unique(labels, return_counts=True)
        min_count = min(counts)
        balanced_data = []
        balanced_labels = []
        for label in unique:
            label_data = data[labels == label]
            label_data_resampled = resample(label_data, replace=False, n_samples=min_count, random_state=42)
            balanced_data.append(label_data_resampled)
            balanced_labels.extend([label] * min_count)
        balanced_data = np.vstack(balanced_data)
        balanced_labels = np.array(balanced_labels)
        return balanced_data, balanced_labels

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
    print(f"Loaded data from {file_path_X} and {file_path_y}")
    print(f"Data shape: {dataset.data.shape}, Labels shape: {dataset.labels.shape}")
    return dataloader
