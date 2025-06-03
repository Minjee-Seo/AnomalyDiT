# from asyncio import constants
# from typing import Any
# import torch
# from dataset import *
# from visualize import *
# from anomaly_map import *
# from metrics import *
# from reconstruction import *
# os.environ['CUDA_VISIBLE_DEVICES'] = "0,1,2"

import os
import torch
from typing import Any
from dataset import *
from reconstruction import *
from metrics import *
from visualize import *
from anomaly_map import *

class DDAD:
    def __init__(self, dit_model, config) -> None:
        self.test_dataset = Dataset_maker(
            root=config.data.data_dir,
            category=config.data.category,
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
                x = x.to(self.config.model.device)  # (B, 1, T)
                target = target.to(self.config.model.device)
                label_tensor = torch.tensor(
                    [0 if l == 'good' else 1 for l in label],
                    dtype=torch.long, device=self.config.model.device
                )

                # 복원된 시계열 생성 (x0)
                x0 = self.reconstruction(x, y=label_tensor)[-1]  # (B, 1, T)

                # 이상 점수 계산 (MSE)
                score = torch.mean(torch.abs(x0 - x), dim=2)  # (B, 1)
                predictions.extend(score.squeeze(1).cpu().tolist())
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

        # 결과 시각화
        reconstructed_list = torch.cat(reconstructed_list, dim=0)  # (N, 1, T)
        forward_list = torch.cat(forward_list, dim=0)              # (N, 1, T)
        gt_list = torch.cat(gt_list, dim=0)                        # (N, 1, T)
        pred_scores = torch.tensor(predictions).unsqueeze(1)      # (N, 1)
        pred_mask = (pred_scores > metric.threshold).float()      # (N, 1)

        if not os.path.exists('results'):
            os.mkdir('results')

        if self.config.metrics.visualisation:
            visualize(
                forward_list,
                reconstructed_list,
                gt_list,
                pred_mask,
                pred_scores,
                self.config.data.category
            )