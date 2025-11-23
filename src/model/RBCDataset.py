


import torch
from torch import nn
from torch.utils.data import Dataset
import numpy as np
import mysql.connector

class RBCDatasetDB(Dataset):
    def __init__(self, db_config, use_log_image=False, transform=None):
        self.use_log_image = use_log_image
        self.transform = transform

        # Connect to the DB and load all rows into memory
        self.conn = mysql.connector.connect(**db_config)
        self.cursor = self.conn.cursor(dictionary=True)

        self.cursor.execute("SELECT regular_image, log_image, diameter, thickness, ratio, ref_index FROM ImageData")
        self.data = self.cursor.fetchall()

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        row = self.data[idx]

        img_text = row['log_image'] if self.use_log_image else row['regular_image']
        values = [float(line.replace('D', 'E')) for line in img_text.strip().splitlines()]
        image = np.array(values, dtype=np.float32).reshape((50, 50))
        image = torch.from_numpy(image).unsqueeze(0)  # shape: [1, 50, 50]

        if self.transform:
            image = self.transform(image)

        target = torch.tensor([
            row['diameter'],
            row['thickness'],
            row['ratio'],
            row['ref_index']
        ], dtype=torch.float32)

        return image, target

    def __del__(self):
        try:
            cur = getattr(self, "_cursor", None)
            if cur is not None:
                try:
                    cur.close()
                except Exception:
                    pass
            self._cursor = None

            conn = getattr(self, "_conn", None)
            if conn is not None:
                try:
                    conn.close()
                except Exception:
                    pass
            self._conn = None
        except Exception:
            # swallow anything at interpreter shutdown
            pass

class DenoisingDataset(torch.utils.data.Dataset):
    def __init__(self, base_dataset, noise_transform: nn.Module):
        super().__init__()
        self.base_dataset = base_dataset
        self.noise_transform = noise_transform

    def __len__(self):
        return len(self.base_dataset)

    def __getitem__(self, idx):
        x, _ = self.base_dataset[idx]
        x_noisy = self.noise_transform(x)
        return x_noisy, x

