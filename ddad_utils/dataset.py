import os
import numpy as np
import pandas as pd
# from glob import glob
# from PIL import Image
import torch
from torch.utils.data import Dataset
from torchvision import transforms


class Dataset_maker(Dataset):
    def __init__(self, root, config, is_train=True):
        self.config = config
        self.is_train = is_train

        if is_train:
            self.data = pd.read_csv(os.path.join(root, "ECG_Train_with_normal.csv")).values
            self.labels = np.zeros(len(self.data))  # 정상만 있으므로 라벨 0
        else:
            self.data = pd.read_csv(os.path.join(root, "ECG_Test_with_anomaly.csv")).values
            label_path = os.path.join(root, "ECG_Anomaly_PointLabels.npy")
            self.labels = np.load(label_path)  # shape: (length,)
        
        # 시계열 윈도우 생성
        self.window_size = config.data.window_size
        self.stride = config.data.stride

        self.samples = []
        for i in range(0, len(self.data) - self.window_size + 1, self.stride):
            x = self.data[i:i+self.window_size]
            if not is_train:
                y = self.labels[i:i+self.window_size]
                self.samples.append((x, y))
            else:
                self.samples.append((x, 0))

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        x, y = self.samples[idx]
        x = torch.tensor(x, dtype=torch.float).unsqueeze(0)  # (1, window_size)
        if self.is_train:
            return x, 'good'
        else:
            y = torch.tensor(y, dtype=torch.float).unsqueeze(0)  # (1, window_size)
            label = 'defective' if y.sum() > 0 else 'good'
            return x, y, label