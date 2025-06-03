from typing import Any
import torch
# from forward_process import *
import numpy as np
import os
from models import *

os.environ['CUDA_VISIBLE_DEVICES'] = "0,1,2"

class Reconstruction:
    """
    시계열 DDPM 기반 복원 클래스 (DiT 기반)
    """

    def __init__(self, model, config) -> None:
        self.model = model
        self.config = config
        self.device = config.model.device

        self.diffusion = create_diffusion(
            timestep_respacing="1000",
            noise_schedule=config.model.noise_schedule,
            diffusion_steps=config.model.trajectory_steps,
        )

    def __call__(self, x, y=None, w=1.0) -> Any:
        """
        조건부 복원 과정 수행
        :param x: 실제 입력 시계열 데이터 (shape: B x 1 x T)
        :param y: 클래스 정보 (DiT 조건부 입력용)
        :param w: 조건 강조 weight (사용하지 않음)
        :return: 전체 trajectory (특히 마지막 값이 복원된 x₀)
        """
        x_t = torch.randn_like(x).to(self.device)  # 랜덤 노이즈에서 시작
        y = y.to(self.device) if y is not None and isinstance(y, torch.Tensor) else None

        all_xt = [x_t]

        with torch.no_grad():
            for t in reversed(range(self.diffusion.num_timesteps)):
                t_batch = torch.full((x.size(0),), t, device=self.device, dtype=torch.long)
                out = self.diffusion.p_sample(
                    self.model,
                    x_t,
                    t_batch,
                    model_kwargs={"y": y} if y is not None else {},
                )
                x_t = out["sample"]
                all_xt.append(x_t)

        return all_xt  # 최종 복원값은 all_xt[-1]

         



