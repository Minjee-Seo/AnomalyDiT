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


def train(model, diffusion, sampler, dataloader, config):
    model.train()
    optimizer = optim.Adam(model.parameters(), lr=config.model.learning_rate)

    best_loss = float('inf')
    best_model_path = ""

    for epoch in range(config.model.epochs):
        print(f"\n[Epoch {epoch+1}/{config.model.epochs}]")
        total_loss = 0

        for i, batch in enumerate(tqdm(dataloader, desc=f"Epoch {epoch+1}", leave=True)):
            if hasattr(config, "training") and config.training.use_label:
                x, label = batch
            else:
                x, _ = batch

            x = x.to(config.model.device)
            y = x[:, 0:1, :].repeat(1, config.data.input_channel, 1).to(config.model.device)

            t, _ = sampler.sample(x.shape[0], config.model.device)
            loss_dict = diffusion.training_losses(model, x, t, model_kwargs={"y": y})
            loss = loss_dict["loss"].mean()

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            total_loss += loss.item()

        avg_loss = total_loss / len(dataloader)
        print(f"Epoch {epoch+1}: Avg Loss = {avg_loss:.4f}")

        # 모델 저장
        if config.model.save_model:
            save_dir = os.path.join(config.model.checkpoint_dir, config.model.exp_name)
            os.makedirs(save_dir, exist_ok=True)
            save_path = os.path.join(save_dir, f"{config.model.checkpoint_name}_epoch{epoch+1}.pt")
            torch.save(model.state_dict(), save_path)
            print(f"모델 저장됨: {save_path}")

            # 최적 모델 저장
            if avg_loss < best_loss:
                best_loss = avg_loss
                best_model_path = os.path.join(save_dir, f"{config.model.checkpoint_name}_best.pt")
                torch.save(model.state_dict(), best_model_path)
                print(f"현재까지 최고 모델 저장됨 (loss={best_loss:.4f}): {best_model_path}")

def main():
    # 1. config.yaml 불러오기
    config = OmegaConf.load("C:/Users/Pro/Desktop/AnomalyDiT-main/AnomalyDiT-main/ddad_utils/config.yaml")

    # 2. 디바이스 설정 및 시드 고정
    device = torch.device(config.model.device)
    torch.manual_seed(config.model.seed)

    # 3. 학습용 Dataset & DataLoader
    train_dataset = Dataset_maker(
        root=config.data.data_dir,
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
        input_size=config.data.seq_len,
        patch_size=5,
        in_channels=config.data.input_channel,
        num_classes=2  # 실제 분류 목적은 없지만 구조상 필요
    ).to(device)

    # 5. pretrained weight 로딩
    if config.model.load_chp > 0:
        load_path = os.path.join(config.model.checkpoint_dir, config.model.checkpoint_name + ".pt")
        if os.path.exists(load_path):
            print(f"Pretrained weight 로드 중: {load_path}")
            state_dict = torch.load(load_path, map_location=device)
            try:
                model.load_state_dict(state_dict, strict=True)
                print("weight 로딩 완료 (strict=True)")
            except RuntimeError as e:
                print("weight 로딩 실패 (구조 불일치):", e)
        else:
            print(f"지정한 weight 파일이 존재하지 않음: {load_path}")

    # 6. Diffusion & Sampling 전략
    diffusion = create_diffusion(
        timestep_respacing="1000",
        noise_schedule=config.model.noise_schedule,
        diffusion_steps=config.model.trajectory_steps,
    )
    sampler = create_named_schedule_sampler("uniform", diffusion)

    # 7. 학습 시작
    train(model, diffusion, sampler, train_loader, config)


if __name__ == "__main__":
    main()
