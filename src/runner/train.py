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
        generator = torch.Generator()
        generator.manual_seed(42)
        return DataLoader(dataset = dataset, batch_size=self.training_config.batch_size, shuffle=True, collate_fn=self.collate_fn, generator=generator)   
        

    def train(self):
        """训练数据"""

        # 加载检查点
        self._load_checkpoint()
        current_step = 0
        dataloader = self._get_dataloader(self.train_dataset)
        for epoch in range(1,1+self.training_config.epochs):
            print(f"epoch {epoch}/{self.training_config.eporchs}")
            for inputs in tqdm(dataloader, desc=f"Training Epoch {epoch}"):
                loss = self.train_one_epoch(inputs)
                current_step += 1
                self.step += 1
                # tensorboard记录loss
                self.writer.add_scalar('train/loss', loss, current_step)

                # 保存检查点
                if current_step % self.training_config.save_steps == 0:
                    self.save_checkpoint()
            

    
    def train_one_epoch(self, inputs):
        """训练每个Batch"""
        self.model.train()
        loss = 0.0
        inputs = {k:v.to(self.device) for k,v in inputs.items()}
        # 添加混合精度
        with torch.autocast(device_type='cuda', enabled=self.training_config.use_amp):
            outputs = self.model(**inputs)
            loss = outputs.loss
        # 反向传播
        self.scaler.scale(loss).backward()
        # 更新参数
        self.scaler.step(self.optimizer)
        self.scaler.update()
        self.optimizer.zero_grad()
        return loss.item()

    def save_checkpoint(self):
        """保存检查点"""
        checkpoint_path = Path(self.training_config.output_dir) / 'checkpoint.pt'
        checkpoint = {
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'step': self.step,
            'early_stop_score': self.early_stop_score,
            'early_stop_counter': self.early_stop_counter,
            'scaler_state_dict': self.scaler.state_dict()
        }
        torch.save(checkpoint, checkpoint_path)
        print(f"Saved checkpoint to {checkpoint_path}")   

    def _load_checkpoint(self):
        """加载检查点"""
        checkpoint_path = Path(self.training_config.output_dir) / 'checkpoint.pt'
        if checkpoint_path.exists():
            checkpoint = torch.load(checkpoint_path)
            self.model.load_state_dict(checkpoint['model_state_dict'])
            self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
            self.step = checkpoint['step']
            self.early_stop_score = checkpoint['early_stop_score']
            self.early_stop_counter = checkpoint['early_stop_counter']
            self.scaler.load_state_dict(checkpoint['scaler_state_dict'])
            print(f"Loaded checkpoint from {checkpoint_path}")
        else:
            print(f"No checkpoint found at {checkpoint_path}, starting from scratch.")


