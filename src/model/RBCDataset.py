import torch
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from PIL import Image
import pandas as pd
import os
class RBCDataset(Dataset):
    def __init__(self, csv_file, image_dir, transform=None):
        self.data = pd.read_csv(csv_file)
        self.image_dir = image_dir
        self.transform = transform

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        row = self.data.iloc[idx]
        img_path = os.path.join(self.image_dir, row['filename'])
        image = Image.open(img_path).convert('L')  # Grayscale

        if self.transform:
            image = self.transform(image)

        # Convert target to tensor: width, depth, hemoglobin
        target = torch.tensor([row['width'], row['depth'], row['hemoglobin']], dtype=torch.float32)
        return image, target
