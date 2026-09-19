import time
from dataclasses import dataclass
from pathlib import Path

import torch

from sklearn.metrics import accuracy_score , f1_score
from torch.utils.checkpoint import checkpoint
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm
from transformers import AutoTokenizer , AutoModelForSequenceClassification , DataCollatorWithPadding

from configuration import config
from preprocess import dataset


@dataclass
class TrainingConfig:
    epochs : int = 10
    batch_size: int = 16
    learning_rate : float = 5e-5
    output_dir: str = './models'
    log_dir :str = './logs'
    save_steps: int = 100
    early_stop_metric : str = 'loss'
    early_stop_patience: int = 3
    use_amp: bool = True

class train:
    """训练类"""
    def __init__(self,device,
                 model,
                 valid_dataset,
                 collate_fn,
                 compute_metrics,
                 train_dataset=None,
                 training_config=TrainingConfig()):
        # 训练的超参数
        self.training_config = training_config

        # 设置设备和模型
        self.device = device
        self.model = model
        # 数据集和数据集整理器
        self.collate_fn = collate_fn
        self.compute_metrics = compute_metrics
        self.train_dataset = train_dataset
        self.valid_dataset = valid_dataset

        # 优化器
        self.optimizer = torch.optim.Adam(model.parameters(), lr=training_config.learning_rate)

        # 评估函数
        self.compute_metrics = compute_metrics

        # 全局的step
        self.step = 0
        # tensorboard
        self.writer = SummaryWriter(log_dir=training_config.log_dir) / time.strftime("%Y%m%d-%H%M%S")

        # 早停
        self.early_stop_score = -float('inf')
        self.early_stop_counter = 0

        # amp
        self.scaler = torch.cuda.amp.GradScaler('cuda', enabled=training_config.use_amp)


    def _get_dataloader(self , dataset):
        """得到dataloader"""
        dataset.set_format(ytpe='torch')
        




