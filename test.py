import torch
import os
import pandas as pd
import numpy as np
from omegaconf import OmegaConf
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

    best_model_path = os.path.join(config.model.checkpoint_dir, config.model.exp_name, f"{config.model.checkpoint_name}_best.pt")
    model.load_state_dict(torch.load(best_model_path, map_location=device))
    model.eval()

    pred_scores = []

    ddad = DDAD(model, config)
    ddad()