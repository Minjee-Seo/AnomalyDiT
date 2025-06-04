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


def train(model, diffusion, sampler, dataloader, config):
    model.train()
    optimizer = optim.Adam(model.parameters(), lr=config.model.learning_rate)

    best_loss = float('inf')
    best_model_path = ""

    for epoch in range(config.model.epochs):
        print(f"\n[Epoch {epoch+1}/{config.model.epochs}]")
        total_loss = 0

        for i, batch in enumerate(tqdm(dataloader, desc=f"Epoch {epoch+1}", leave=True)):
            if config.training.use_label:
                x, _ = batch  # label은 reconstruction에 사용하지 않음
            else:
                x = batch

            x = x.to(config.model.device)

            # condition y는 input signal의 첫 번째 채널
            y = x[:, 0:1, :].repeat(1, config.data.input_channel, 1).to(config.model.device)

            # diffusion timestep 샘플링
            t, _ = sampler.sample(x.shape[0], config.model.device)

            # 모델 학습
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

            # 최적 모델 따로 저장
            if avg_loss < best_loss:
                best_loss = avg_loss
                best_model_path = os.path.join(save_dir, f"{config.model.checkpoint_name}_best.pt")
                torch.save(model.state_dict(), best_model_path)
                print(f"최고 성능 모델 갱신 (loss={best_loss:.4f}): {best_model_path}")


def main():
    # 1. config.yaml 불러오기
    config = OmegaConf.load("C:/Users/Pro/Desktop/AnomalyDiT-main/AnomalyDiT-main/ddad_utils/config.yaml")

    # 2. 디바이스 설정 및 시드 고정
    device = torch.device(config.model.device)
    torch.manual_seed(config.model.seed)

    # 3. 학습용 Dataset & DataLoader 구성
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
        num_classes=2  # reconstruction 목적이지만 구조상 필요
    ).to(device)

    # 5. pretrained weight 불러오기
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
            print(f"weight 파일 없음: {load_path}")

    # 6. Diffusion & Sampler 구성
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
