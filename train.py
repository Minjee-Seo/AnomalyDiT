import os
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from omegaconf import OmegaConf
from models.dit import DiT
from models import create_diffusion
from models.timestep_sampler import create_named_schedule_sampler
from ddad_utils.dataset import Dataset_maker
from tqdm import tqdm
import time
import sys


def train(model, diffusion, sampler, dataloader, config):
    model.train()
    optimizer = optim.Adam(model.parameters(), lr=config.model.learning_rate)

    for epoch in range(config.model.epochs):
        print(f"\n[Epoch {epoch+1}/{config.model.epochs}]")
        total_loss = 0

        for i, batch in enumerate(tqdm(dataloader, desc=f"Epoch {epoch+1}", leave=True)):
            x, label = batch
            x = x.to(config.model.device)
            label = label.to(config.model.device)

            t, _ = sampler.sample(x.shape[0], config.model.device)
            loss_dict = diffusion.training_losses(model, x, t, model_kwargs={"y": label})
            loss = loss_dict["loss"].mean()

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            total_loss += loss.item()

        avg_loss = total_loss / len(dataloader)
        print(f"Epoch {epoch+1}: Avg Loss = {avg_loss:.4f}")


def main():
    # 1. config.yaml 불러오기
    config = OmegaConf.load("config.yaml")

    # 2. 디바이스 설정 및 시드 고정
    device = torch.device(config.model.device)
    torch.manual_seed(config.model.seed)

    # 3. 학습용 Dataset & DataLoader
    train_dataset = Dataset_maker(
        root=config.data.data_dir,
        category=config.data.category,
        config=config,
        is_train=True,
    )
    train_loader = DataLoader(
        train_dataset,
        batch_size=config.data.batch_size,
        shuffle=True,
        num_workers=config.model.num_workers,
    )

    # 4. DiT 모델 초기화
    model = DiT(
        input_size=750,
        patch_size=5,
        in_channels=config.data.input_channel,
        num_classes=42  # 시계열 클래스 수 (예: 환자 ID)
    ).to(device)

    # 5. Diffusion & Sampling 전략
    diffusion = create_diffusion(
        timestep_respacing="1000",
        noise_schedule=config.model.noise_schedule,
        diffusion_steps=config.model.trajectory_steps,
    )
    sampler = create_named_schedule_sampler("uniform", diffusion)

    # 6. 학습
    train(model, diffusion, sampler, train_loader, config)

    # 7. 모델 저장
    if config.model.save_model:
        os.makedirs(config.model.checkpoint_dir, exist_ok=True)
        save_path = os.path.join(config.model.checkpoint_dir, config.model.checkpoint_name + ".pt")
        torch.save(model.state_dict(), save_path)
        print(f"\n 모델 저장 완료: {save_path}")


if __name__ == "__main__":
    main()





# import os
# import torch
# from torch.utils.data import DataLoader, Dataset
# import torch.nn as nn
# import torch.optim as optim
# import numpy as np
# import pandas as pd
# from models.dit import DiT
# from models import create_diffusion
# from models.timestep_sampler import create_named_schedule_sampler


# class TimeSeriesDataset(Dataset):
#     def __init__(self, csv_path):
#         df = pd.read_csv(csv_path, header=None)

#         # 문자열 레이블을 정수로 매핑
#         label_map = {label: idx for idx, label in enumerate(df.iloc[:, 0].unique())}
#         df.iloc[:, 0] = df.iloc[:, 0].map(label_map)

#         self.X = torch.tensor(df.iloc[:, 1:].values, dtype=torch.float32).unsqueeze(1)  # (B, 1, T)
#         self.y = torch.tensor(df.iloc[:, 0].values, dtype=torch.long)  # 정수 레이블

#     def __len__(self):
#         return len(self.X)

#     def __getitem__(self, idx):
#         return self.X[idx], self.y[idx]


# def train(model, diffusion, schedule_sampler, dataloader, device, epochs=10):
#     model.train()
#     optimizer = optim.Adam(model.parameters(), lr=1e-4)

#     for epoch in range(epochs):
#         print(f"\n[Epoch {epoch+1}/{epochs}]")
#         total_loss = 0
#         for i, batch in enumerate(dataloader):
#             x, y = batch
#             x = x.to(device)
#             y = y.to(device)
#             t, _ = schedule_sampler.sample(x.shape[0], device)
#             # t = torch.randint(0, diffusion.num_timesteps, (x.shape[0],), device=device).long()
#             loss_dict = diffusion.training_losses(model, x, t, model_kwargs={'y': y})
#             loss = loss_dict['loss'].mean()

#             optimizer.zero_grad()
#             loss.backward()
#             optimizer.step()

#             total_loss += loss.item()
#             if i % 10 == 0:  # 10개마다만 출력
#                 print(f"  Step {i+1}/{len(dataloader)} - Loss: {loss.item():.4f}")

#         avg_loss = total_loss / len(dataloader)
#         print(f"Epoch {epoch+1}: Avg Loss = {avg_loss:.4f}")


# def main():
#     # 환경 설정
#     device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

#     # 데이터셋 로딩
#     train_dataset = TimeSeriesDataset('C:/Users/Pro/Desktop/AnomalyDiT-main/AnomalyDiT-main/Dataset/ECG_Train_with_normal.csv')
#     train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True)

#     # 클래스 수 정의
#     num_classes = 42

#     # 모델 및 확산 생성
#     model = DiT(input_size=750, patch_size=5, in_channels=1, num_classes=num_classes).to(device)
#     diffusion = create_diffusion(timestep_respacing="1000")
#     schedule_sampler = create_named_schedule_sampler("uniform", diffusion)

#     # 학습 시작
#     train(model, diffusion, schedule_sampler, train_loader, device, epochs=20)

#     # 모델 저장
#     torch.save(model.state_dict(), 'dit_ts_model.pt')


# if __name__ == '__main__':
#     main()


