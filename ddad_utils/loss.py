import torch
import torch.nn as nn
import numpy as np

def get_loss(model, x_0, t, config, y=None):
    """
    DDPM 기반 MSE loss 계산
    :param model: DiT 모델
    :param x_0: 정답 시계열 (B, C, T)
    :param t: 시간 step 텐서 (B,)
    :param config: 설정(config.model.trajectory_steps 등)
    :param y: 조건 시계열 (B, C_cond, T) 또는 None
    :return: 평균 MSE loss
    """
    device = config.model.device
    x_0 = x_0.to(device)
    t = t.to(device)

    # 선형 베타 스케줄 생성
    betas = np.linspace(
        config.model.beta_start,
        config.model.beta_end,
        config.model.trajectory_steps,
        dtype=np.float64,
    )
    betas = torch.tensor(betas, dtype=torch.float32, device=device)  # (T,)

    # 시간 스텝별 alpha 계산
    alpha_cumprod = (1 - betas).cumprod(dim=0)  # (T,)
    a_t = alpha_cumprod.index_select(0, t).view(-1, 1, 1)  # (B, 1, 1)

    # 노이즈 샘플링
    eps = torch.randn_like(x_0, device=device)
    x_t = torch.sqrt(a_t) * x_0 + torch.sqrt(1 - a_t) * eps  # (B, C, T)

    # 조건 시계열 처리
    model_kwargs = {"y": y.to(device).float()} if y is not None else {}

    # 예측 노이즈
    eps_pred = model(x_t, t, **model_kwargs)  # (B, C, T)

    # MSE loss
    loss = nn.functional.mse_loss(eps_pred, eps)
    return loss
