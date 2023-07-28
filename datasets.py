import os
import torch
import torch.nn as nn
import torchvision.transforms as transforms
from torch.utils.data import Dataset, DataLoader
from PIL import Image

class CustomDataset(Dataset):
    def __init__(self, image_directory, transform=None):
        self.image_directory = image_directory
        self.transform = transform
        self.image_paths, self.labels = self._load_data()

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, index):
        image = Image.open(self.image_paths[index])
        if self.transform:
            image = self.transform(image)

        label = torch.Tensor(self.labels[index])

        return image, label

    def _load_data(self):
        image_paths = []
        labels = []

        for filename in os.listdir(self.image_directory):
            if filename.endswith('.jpg') or filename.endswith('.png'):  # Add appropriate file extensions
                image_path = os.path.join(self.image_directory, filename)
                image_paths.append(image_path)

                image_name = os.path.splitext(filename)[0]
                image_name = image_name.split('_')
                image_name = image_name[image_name.index('hls')+1:]
                label_parts = image_name
                label_parts = [int(part) for part in label_parts]
                normalized_labels = [part / divisor for part, divisor in zip(label_parts, [180, 255, 255,
                                                                                           180, 255, 255])]
                labels.append(normalized_labels)

        return image_paths, labels