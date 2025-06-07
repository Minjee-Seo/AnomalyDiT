import os
from typing import Any
import torch
from models import create_diffusion

# CUDA 디바이스 설정 (멀티 GPU 설정 시 환경 변수 이용 가능)
# os.environ['CUDA_VISIBLE_DEVICES'] = "0,1,2"

class Reconstruction:
    """
    시계열 DDPM 기반 복원 클래스 (DiT 기반)
    """

    def __init__(self, model, config) -> None:
        self.model = model
        self.config = config
        self.device = config.model.device

        # Diffusion 구성
        self.diffusion = create_diffusion(
            timestep_respacing="1000",  # sampling 구간 수
            noise_schedule=config.model.noise_schedule,
            diffusion_steps=config.model.trajectory_steps,
        )

    def __call__(self, x: torch.Tensor, y: torch.Tensor = None, w: float = 1.0) -> Any:
        """
        조건부 복원 과정 수행

        :param x: 실제 입력 시계열 데이터 (B, C, T)
        :param y: degraded signal (조건 입력, B, 1, T)
        :param w: 조건 강조 weight (미사용)
        :return: 전체 복원 trajectory (최종 복원 결과는 all_xt[-1])
        """
        x_t = torch.randn_like(x).to(self.device)  # 랜덤 노이즈 초기화

        

        # all_xt = [x_t]

        with torch.no_grad():
            for t in reversed(range(self.diffusion.num_timesteps)):
                t_batch = torch.full((x.size(0),), t, device=self.device, dtype=torch.long)

                model_kwargs = {"y": y}

                out = self.diffusion.p_sample(
                    self.model,
                    x_t,
                    t_batch,
                    model_kwargs=model_kwargs,
                )
                x_t = out["sample"]
                # all_xt.append(x_t)

        return x_t  # 최종 복원 결과는 all_xt[-1]
