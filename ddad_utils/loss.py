import torch
import torch.nn as nn
import numpy as np

def get_loss(model, x_0, t, config, y=None):
    """
    DDPM 방식의 MSE loss 계산 (시계열 데이터용)
    - model: DiT 모델
    - x_0: 정답 시계열 데이터 (B, 1, T)
    - t: 시간 step 텐서 (B,)
    - config: 설정
    - y: 클래스 레이블 (optional, B,)
    """
    device = config.model.device
    x_0 = x_0.to(device)
    t = t.to(device)

    # 베타 스케줄
    betas = np.linspace(config.model.beta_start, config.model.beta_end, config.model.trajectory_steps, dtype=np.float64)
    b = torch.tensor(betas).type(torch.float32).to(device)

    # 노이즈 추가
    e = torch.randn_like(x_0, device=device)
    at = (1 - b).cumprod(dim=0).index_select(0, t).view(-1, 1, 1)  # 시계열은 (B, 1, T)

    # forward noise x_t 생성
    x_t = at.sqrt() * x_0 + (1 - at).sqrt() * e

    # DiT 예측
    model_kwargs = {"y": y} if y is not None else {}
    e_hat = model(x_t, t, **model_kwargs)  # (B, 1, T)

    # loss 계산 (예측한 노이즈 vs 실제 노이즈)
    loss = (e - e_hat).square().mean()
    return loss
