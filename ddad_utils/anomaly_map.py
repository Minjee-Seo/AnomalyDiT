# import torch
# import torch.nn.functional as F
# from kornia.filters import gaussian_blur2d
# from torchvision.transforms import transforms
# import math 
# from dataset import *
# from visualize import *
# from feature_extractor import *
# import numpy as np


import torch
import torch.nn.functional as F


def heat_map(output, target, FE, config):
    """
    시계열 reconstruction 기반 이상 점수 계산 함수
    - 입력: output (재구성 시계열), target (원본 시계열)
    - 출력: anomaly_map (시점별 reconstruction error 기반 이상 점수)
    - FE는 시계열에서는 사용되지 않음 (None 처리)
    """
    output = output.to(config.model.device)
    target = target.to(config.model.device)

    # reconstruction error를 기반으로 한 이상 점수 (Mean Absolute Error)
    anomaly_map = torch.abs(output - target)  # (B, 1, T)

    return anomaly_map  # (B, 1, T)


def pixel_distance(output, target):
    """
    시계열 각 시점별 절댓값 차이를 반환
    - 입력: output (재구성 시계열), target (원본 시계열)
    - 출력: 차이값 (B, 1, T)
    """
    return torch.abs(output - target)  # (B, 1, T)

