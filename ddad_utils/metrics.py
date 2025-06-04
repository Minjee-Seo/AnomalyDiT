# import torch
# from torchmetrics import ROC, AUROC, F1Score
# import os
# from torchvision.transforms import transforms
# from skimage import measure
# import pandas as pd
# from statistics import mean
# import numpy as np
# from sklearn.metrics import auc
# from sklearn import metrics
# from sklearn.metrics import roc_auc_score, roc_curve, precision_recall_fscore_support

import torch
import numpy as np
from sklearn.metrics import roc_auc_score, roc_curve, precision_recall_fscore_support

class Metric:
    def __init__(self, labels_list, predictions, anomaly_map_list, gt_list, config):
        self.labels_list = labels_list  # [0, 1, 1, 0, ...]
        self.predictions = predictions  # [score1, score2, ...]
        self.config = config
        self.threshold = 0.5

    def image_auroc(self):
        return roc_auc_score(self.labels_list, self.predictions)

    def optimal_threshold(self):
        fpr, tpr, thresholds = roc_curve(self.labels_list, self.predictions)
        youden_j = tpr - fpr
        optimal_threshold_index = np.argmax(youden_j)
        self.threshold = thresholds[optimal_threshold_index]
        return self.threshold

    def precision_recall_f1(self):
        predictions_tensor = torch.tensor(self.predictions)
        labels_tensor = torch.tensor(self.labels_list)
        pred_binary = (predictions_tensor > self.threshold).int()

        precision, recall, f1, _ = precision_recall_fscore_support(
            labels_tensor.cpu().numpy(),
            pred_binary.cpu().numpy(),
            average='binary'
        )
        print(f"Precision: {precision:.4f}")
        print(f"Recall:    {recall:.4f}")
        print(f"F1-Score:  {f1:.4f}")

    def miscalssified(self):
        predictions_tensor = torch.tensor(self.predictions)
        labels_tensor = torch.tensor(self.labels_list)
        pred_binary = (predictions_tensor > self.threshold).int()

        for i, (true_label, pred) in enumerate(zip(labels_tensor, pred_binary)):
            if true_label != pred:
                print(f"[Sample {i}] Predicted: {pred.item()} / True: {true_label.item()} --> INCORRECT")