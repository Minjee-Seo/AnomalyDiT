import torch
import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from torch.utils.data import Dataset, DataLoader
from models.dit import DiT
from models import *
from ddad_utils.dataset import Dataset_maker
from ddad_utils.ddad import DDAD
from tqdm import tqdm
from sklearn.metrics import (
    classification_report,
    precision_recall_curve,
    roc_curve,
    auc
)

if __name__=="__main__":

    # Load DDAD configurations
    config = OmegaConf.load("./ddad_utils/config.yaml")

    device = torch.device(config.model.device)
    torch.manual_seed(config.model.seed)

    os.makedirs('ddad_results', exist_ok=True)

    csv_path = "Dataset/ECG_Test_with_anomaly.csv"
    label_path = "Dataset/ECG_Anomaly_PointLabels.npy"

    # Define dataset and dataloader
    test_dataset = Dataset_maker(
        root='./Dataset',
        config=config,
        is_train=False,
    )
    test_loader = DataLoader(
        test_dataset,
        batch_size=config.data.batch_size,
        shuffle=True,
        num_workers=config.model.num_workers,
    )

    point_labels = np.load(label_path).flatten()  # 시점별 이상 라벨

    model = DiT(
        input_size=config.data.seq_len,
        patch_size=5,
        in_channels=config.data.input_channel,
        num_classes=2
    ).to(device)

    model.load_state_dict(torch.load("dit_ts_model_best.pt", map_location=device))
    model.eval()

    pred_scores = []

    ddad = DDAD(model, config)
    reconstructed_list, forward_list, gt_list, pred_scores, pred_mask = ddad()

    ######################################################################
    ####################### !!! 아래 수정중 !!! ######################
    ######################################################################

    pred_scores = torch.cat(pred_scores, dim=0).numpy().flatten()

    precision, recall, thresholds_pr = precision_recall_curve(point_labels, pred_scores)
    f1_scores = 2 * (precision * recall) / (precision + recall + 1e-8)
    best_idx = np.argmax(f1_scores)
    best_threshold = thresholds_pr[best_idx]

    fpr, tpr, thresholds_roc = roc_curve(point_labels, pred_scores)
    roc_auc = auc(fpr, tpr)

    # pred_binary = (pred_scores > best_threshold).astype(int)
    pred_binary = pred_mask

    print(f"▶ Best Threshold (by F1): {best_threshold:.4f}")
    print(f"▶ Best F1 Score: {f1_scores[best_idx]:.4f}")
    print(f"▶ Precision at best: {precision[best_idx]:.4f}")
    print(f"▶ Recall at best: {recall[best_idx]:.4f}")
    print(f"▶ ROC AUC Score: {roc_auc:.4f}")

    plt.figure(figsize=(8, 4))
    plt.plot(thresholds_pr, f1_scores[1:], label="F1-score")
    plt.axvline(best_threshold, color='red', linestyle='--', label=f"Best Threshold = {best_threshold:.4f}")
    plt.title("F1-score vs Threshold")
    plt.xlabel("Threshold")
    plt.ylabel("F1-score")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(os.path.join('ddad_results','f1_threshold.png'))
    plt.close()

    plt.figure(figsize=(6, 6))
    plt.plot(recall, precision, label='PR Curve')
    plt.scatter(recall[best_idx], precision[best_idx], color='red', label='Best F1 Point')
    plt.xlabel("Recall")
    plt.ylabel("Precision")
    plt.title("Precision-Recall Curve")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(os.path.join('ddad_results','precision_recall_curve.png'))
    plt.close()

    plt.figure(figsize=(6, 6))
    plt.plot(fpr, tpr, label=f"ROC Curve (AUC = {roc_auc:.4f})")
    plt.plot([0, 1], [0, 1], linestyle='--', color='gray')
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.title("ROC Curve")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(os.path.join('ddad_results','ROC_curve.png'))
    plt.close()

    print(f"🔍 F1-score 기준 최적 threshold: {best_threshold:.4f}")
    print(classification_report(point_labels, pred_binary, target_names=["정상", "이상"]))

    plt.figure(figsize=(14, 5))
    plt.plot(pred_scores, label="Predicted Scores", linewidth=1)
    plt.plot(point_labels * 1.0, label="True Anomaly", linestyle='--')  # ✔ 스케일 조절
    plt.axhline(best_threshold, color='red', linestyle=':', label=f"Threshold = {best_threshold:.4f}")
    plt.title("Predicted Scores vs True Anomaly Labels")
    plt.xlabel("Time Point Index")
    plt.ylabel("Score")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(os.path.join('ddad_results','predicted_scores_and_labels.png'))
    plt.close()

    plt.figure(figsize=(8, 4))
    plt.hist(pred_scores, bins=100, color='steelblue')
    plt.axvline(best_threshold, color='red', linestyle='--', label=f"Threshold = {best_threshold:.4f}")
    plt.title("Distribution of Predicted Scores")
    plt.xlabel("Score")
    plt.ylabel("Frequency")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(os.path.join('ddad_results','predicted_scores_distribution.png'))
    plt.close()

    # 시계열 데이터 추출
    X = test_dataset.X.numpy()         # shape: (N, 1, T)
    test_series = X[:, 0, :]           # shape: (N, T)

    # point_labels와 pred_scores reshape
    point_labels = point_labels.reshape(test_series.shape)
    pred_scores = pred_scores.reshape(test_series.shape)

    test_labels = test_dataset.y.numpy()

    i = 0
    signal = test_series[i]
    true_anom = point_labels[i]
    pred_score = pred_scores[i]
    pred_anom = (pred_score > best_threshold).astype(int)

    plt.figure(figsize=(12, 5))
    plt.plot(signal, label="Signal")
    plt.plot(true_anom * np.max(signal), label="True Anomaly", linestyle='--')
    plt.plot(pred_anom * np.max(signal), label="Predicted Anomaly", linestyle=':')
    plt.title(f"Sample {i} - True vs Predicted Anomalies")
    plt.xlabel("Time")
    plt.ylabel("Value")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(os.path.join('ddad_results','true_and_predicted_anomalies.png'))
    plt.close()

    # 고유 클래스 목록
    unique_classes = sorted(np.unique(test_labels))

    # subplot 시각화
    n_rows, n_cols = 6, 7  # 42개 클래스 기준
    plt.figure(figsize=(22, 18))

    for idx, cls in enumerate(unique_classes):
        # 해당 클래스의 첫 번째 시계열 index 찾기
        sample_idx = np.where(test_labels == cls)[0][0]
        
        signal = test_series[sample_idx]
        true_anom = point_labels[sample_idx]
        pred_score = pred_scores[sample_idx]
        pred_anom = (pred_score > best_threshold).astype(int)

        plt.subplot(n_rows, n_cols, idx + 1)
        plt.plot(signal, label='Signal', linewidth=1)
        plt.plot(true_anom * np.max(signal), '--', label='True Anomaly', linewidth=1)
        plt.plot(pred_anom * np.max(signal), ':', label='Predicted Anomaly', linewidth=1)
        plt.title(f"Class {cls+1} (idx {sample_idx})")
        plt.xticks([])
        plt.yticks([])
        plt.grid(True)

    # 전체 제목 및 범례
    plt.suptitle("Class-wise Anomaly Detection: True vs Predicted", fontsize=18, y=0.92)
    plt.tight_layout(rect=[0, 0, 1, 0.94])
    plt.legend(loc='lower right', bbox_to_anchor=(1.15, 0.1))
    plt.savefig(os.path.join('ddad_results','classwise_anomaly_detection.png'))
    plt.close()