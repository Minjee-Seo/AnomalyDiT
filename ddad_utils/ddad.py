import os
import torch
from typing import Any
from ddad_utils.dataset import Dataset_maker
from reconstruction import Reconstruction
from metrics import Metric
from visualize import visualize
from anomaly_map import *

class DDAD:
    def __init__(self, dit_model, config) -> None:
        self.test_dataset = Dataset_maker(
            root=config.data.data_dir,
            config=config,
            is_train=False,
        )
        self.testloader = torch.utils.data.DataLoader(
            self.test_dataset,
            batch_size=config.data.test_batch_size,
            shuffle=False,
            num_workers=config.model.num_workers,
            drop_last=False,
        )
        self.dit_model = dit_model
        self.config = config
        self.reconstruction = Reconstruction(self.dit_model, self.config)

    def __call__(self) -> Any:
        labels_list = []
        predictions = []
        gt_list = []
        reconstructed_list = []
        forward_list = []

        with torch.no_grad():
            for x, target, label in self.testloader:
                x = x.to(self.config.model.device)               # (B, C, T)
                target = target.to(self.config.model.device)     # (B, 1, T)

                # label: 정상=0, 이상=1 (정수형)
                label_tensor = label.to(self.config.model.device)

                # 조건 입력 y는 reconstruction 내부에서 자동 지정되도록
                # x0 = self.reconstruction(x)[-1].detach()         # (B, C, T)
                degraded_y = x[:, 0:1, :]
                x0 = self.reconstruction(x, y=degraded_y)[-1].detach()

                # 이상 점수 계산 (MSE 또는 MAE 가능)
                score = torch.mean(torch.abs(x0 - x), dim=(1, 2))  # (B,)
                predictions.extend(score.cpu().tolist())
                labels_list.extend(label_tensor.cpu().tolist())

                gt_list.append(target)
                forward_list.append(x)
                reconstructed_list.append(x0)

        # 성능 평가
        metric = Metric(labels_list, predictions, None, gt_list, self.config)
        metric.optimal_threshold()

        print("\n[Evaluation Metrics]")
        if self.config.metrics.auroc:
            print('AUROC: ({:.1f}, -)'.format(metric.image_auroc() * 100))
        if self.config.metrics.pro:
            print('PRO: {:.1f}'.format(metric.pixel_pro() * 100))
        if self.config.metrics.misclassifications:
            metric.miscalssified()
        metric.precision_recall_f1()

        # 시각화
        reconstructed_list = torch.cat(reconstructed_list, dim=0)  # (N, C, T)
        forward_list = torch.cat(forward_list, dim=0)              # (N, C, T)
        gt_list = torch.cat(gt_list, dim=0)                        # (N, 1, T)
        pred_scores = torch.tensor(predictions).unsqueeze(1)      # (N, 1)
        pred_mask = (pred_scores > metric.threshold).float()      # (N, 1)

        os.makedirs('results', exist_ok=True)

        if self.config.metrics.visualisation:
            visualize(
                forward_list,
                reconstructed_list,
                gt_list,
                pred_mask,
                pred_scores,
                self.config.data.category
            )
