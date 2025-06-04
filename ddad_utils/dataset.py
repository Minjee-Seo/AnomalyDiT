import os
import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset

class Dataset_maker(Dataset):
    def __init__(self, root, config, is_train=True):
        self.config = config
        self.is_train = is_train

        if is_train:
            # 학습 데이터 로드
            df = pd.read_csv(os.path.join(root, "ECG_Train_with_normal.csv"))
            df = df.apply(pd.to_numeric, errors='coerce').fillna(0)

            self.labels = df["label"].astype(int).values
            self.X = df.drop(columns=["label"]).astype(np.float32).values

        else:
            # 테스트 데이터 로드
            df = pd.read_csv(os.path.join(root, "ECG_Test_with_anomaly.csv"))
            df = df.apply(pd.to_numeric, errors='coerce').fillna(0)

            self.labels = df["label"].astype(int).values
            self.X = df.drop(
                columns=["label", "condition"] + [f"anomaly_{i}" for i in range(config.data.seq_len)]
            ).astype(np.float32).values

            self.anomaly_mask = df[
                [f"anomaly_{i}" for i in range(config.data.seq_len)]
            ].astype(np.float32).values

        # 입력 차원 설정
        self.channels = config.data.input_channel
        self.seq_len = config.data.seq_len
        self.X = self.X.reshape(-1, self.channels, self.seq_len)

    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx):
        x = torch.tensor(self.X[idx], dtype=torch.float32)
        label = torch.tensor(self.labels[idx], dtype=torch.long)

        if self.is_train:
            return x, label
        else:
            target = torch.tensor(self.anomaly_mask[idx], dtype=torch.float32).unsqueeze(0)  # (1, T)
            return x, target, label
