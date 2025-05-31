import os
import torch
from torch.utils.data import DataLoader, Dataset
import torch.nn as nn
import torch.optim as optim
import numpy as np
import pandas as pd
from models.dit import DiT
from models import create_diffusion


class TimeSeriesDataset(Dataset):
    def __init__(self, csv_path):
        df = pd.read_csv(csv_path, header=None)

        # 문자열 레이블을 정수로 매핑
        label_map = {label: idx for idx, label in enumerate(df.iloc[:, 0].unique())}
        df.iloc[:, 0] = df.iloc[:, 0].map(label_map)

        self.X = torch.tensor(df.iloc[:, 1:].values, dtype=torch.float32).unsqueeze(1)  # (B, 1, T)
        self.y = torch.tensor(df.iloc[:, 0].values, dtype=torch.long)  # 정수 레이블

    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx):
        return self.X[idx], self.y[idx]


def train(model, diffusion, dataloader, device, epochs=10):
    model.train()
    optimizer = optim.Adam(model.parameters(), lr=1e-4)

    for epoch in range(epochs):
        print(f"\n[Epoch {epoch+1}/{epochs}]")
        total_loss = 0
        for i, batch in enumerate(dataloader):
            x, y = batch
            x = x.to(device)
            y = y.to(device)
            t = torch.randint(0, diffusion.num_timesteps, (x.shape[0],), device=device).long()
            loss_dict = diffusion.training_losses(model, x, t, model_kwargs={'y': y})
            loss = loss_dict['loss'].mean()

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            total_loss += loss.item()
            if i % 10 == 0:  # 10개마다만 출력
                print(f"  Step {i+1}/{len(dataloader)} - Loss: {loss.item():.4f}")

        avg_loss = total_loss / len(dataloader)
        print(f"Epoch {epoch+1}: Avg Loss = {avg_loss:.4f}")


def main():
    # 환경 설정
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # 데이터셋 로딩
    train_dataset = TimeSeriesDataset('C:/Users/Pro/Desktop/AnomalyDiT-main/AnomalyDiT-main/Dataset/ECG_Train_with_normal.csv')
    train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True)

    # 클래스 수 정의
    num_classes = 42

    # 모델 및 확산 생성
    model = DiT(input_size=750, patch_size=5, in_channels=1, num_classes=num_classes).to(device)
    diffusion = create_diffusion(timestep_respacing="1000")

    # 학습 시작
    train(model, diffusion, train_loader, device, epochs=20)

    # 모델 저장
    torch.save(model.state_dict(), 'dit_ts_model.pt')


if __name__ == '__main__':
    main()