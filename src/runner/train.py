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
from preprocess.dataset import get_dataset
import json


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

class Trainer:
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
        self.model = model.to(device)
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
        run_dir = Path(training_config.log_dir) / time.strftime("%Y%m%d-%H%M%S")
        self.writer = SummaryWriter(log_dir=str(run_dir))

        # 早停
        self.early_stop_score = -float('inf')
        self.early_stop_counter = 0

        # amp
        self.amp_enabled = (
                training_config.use_amp and self.device.type == 'cuda'
        )

        self.scaler = torch.amp.GradScaler(
            device=self.device.type,
            enabled=self.amp_enabled,
        )

    def _get_dataloader(self , dataset):
        """得到dataloader"""
        dataset.set_format(type='torch')
        generator = torch.Generator()
        generator.manual_seed(42)
        return DataLoader(dataset = dataset, batch_size=self.training_config.batch_size, shuffle=True, collate_fn=self.collate_fn, generator=generator)

    def _should_stop(self, metrics) -> bool:
        """判断是否早停"""
        metric = metrics[self.training_config.early_stop_metric]
        score = -metric if self.training_config.early_stop_metric == 'loss' else metric
        if score > self.early_stop_score:
            self.early_stop_score = score
            self.early_stop_counter = 0
            tqdm.write('保存模型')
            self.model.save_pretrained(str(Path(self.training_config.output_dir) / 'best'))
            return False
        else:
            self.early_stop_counter += 1
            if self.early_stop_counter >= self.training_config.early_stop_patience:
                return True
            else:
                return False

    def train(self):
        """训练数据"""

        # 加载检查点
        self._load_checkpoint()
        current_step = 0
        dataloader = self._get_dataloader(self.train_dataset)
        for epoch in range(1,1+self.training_config.epochs):
            print(f"epoch {epoch}/{self.training_config.epochs}")
            for inputs in tqdm(dataloader, desc=f"Training Epoch {epoch}"):
                loss = self.train_one_epoch(inputs)
                current_step += 1
                # tensorboard记录loss
                self.writer.add_scalar('train/loss', loss, current_step)
                # 保存检查点
                if current_step % self.training_config.save_steps == 0:
                    self.save_checkpoint()
            # 验证评估
            metrics = self.evaluate()
            metrics_str = '| '.join([f"{k}: {v:.4f}" for k, v in metrics.items()])
            tqdm.write(f"Step {current_step} | {metrics_str}")
            # 早停
            if self._should_stop(metrics):
                tqdm.write("早停")
                break
            self.step += 1

    
    def train_one_epoch(self, inputs):
        """训练每个Batch"""
        self.model.train()
        loss = 0.0
        inputs = {k:v.to(self.device) for k,v in inputs.items()}
        # 添加混合精度
        with torch.autocast(
                device_type=self.device.type,
                enabled=self.amp_enabled,
        ):
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
        checkpoint_path.parent.mkdir(parents=True, exist_ok=True)
        checkpoint = {
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'step': self.step,
            'early_stop_score': self.early_stop_score,
            'early_stop_counter': self.early_stop_counter,
            'scaler_state_dict': self.scaler.state_dict()
        }
        # 使用文件对象，避免 torch 在 Windows 下处理中文路径失败
        with checkpoint_path.open("wb") as file:
            torch.save(checkpoint, file)

        print(f"Saved checkpoint to {checkpoint_path}")   

    def _load_checkpoint(self):
        """加载检查点"""
        checkpoint_path = Path(self.training_config.output_dir) / 'checkpoint.pt'
        if not checkpoint_path.exists():
            print(f"No checkpoint found at {checkpoint_path}, starting from scratch.")
            return
        with checkpoint_path.open("rb") as file:
            checkpoint = torch.load(file, map_location=self.device)
        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        self.step = checkpoint['step']
        self.early_stop_score = checkpoint['early_stop_score']
        self.early_stop_counter = checkpoint['early_stop_counter']
        self.scaler.load_state_dict(checkpoint['scaler_state_dict'])
        print(f"Loaded checkpoint from {checkpoint_path}")




    def evaluate(self):
        """评估模型"""
        self.model.eval()
        all_preds = []
        all_labels = []
        total_loss = 0.0
        dataloader = self._get_dataloader(self.valid_dataset)
        for inputs in tqdm(dataloader, desc="Evaluating"):
            inputs = {k:v.to(self.device) for k,v in inputs.items()}
            with torch.no_grad():
                outputs = self.model(**inputs)
                loss = outputs.loss
                total_loss += loss.item()
                # 预测结果
                preds = torch.argmax(outputs.logits, dim=-1)
                all_preds.extend(preds.cpu().numpy())
                all_labels.extend(inputs['labels'].cpu().numpy())
        loss = total_loss / len(dataloader)

        # 计算评估指标
        metrics = self.compute_metrics(all_preds, all_labels)
        return{'loss': loss, **metrics}

def train():
    # devic
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # 分词器
    tokenizer = AutoTokenizer.from_pretrained(config.model_name)

    # 数据集
    train_dataset = get_dataset('train')
    valid_dataset = get_dataset('valid')
    collate_fn = DataCollatorWithPadding(tokenizer=tokenizer,padding = True, return_tensors='pt')
    # 模型
    with open(config.PROCESSED_DATA_DIR / 'labels.json','r',encoding='utf-8') as f:
        all_labels = json.load(f)

    # 模型
    id2label = {index:label for index, label in enumerate(all_labels)}
    label2id = {label:index for index, label in enumerate(all_labels)}

    model = AutoModelForSequenceClassification.from_pretrained(config.model_name, num_labels=len(all_labels), id2label=id2label, label2id=label2id)

    def compute_metrics( preds, labels)->dict:
        """计算评估指标"""
        accuracy = accuracy_score(labels, preds)
        f1 = f1_score(labels, preds, average='weighted')
        return {'accuracy': accuracy, 'f1': f1}
    training_config = TrainingConfig(
        output_dir= config.MODELS_DIR,log_dir=config.LOG_DIR
    )
    trainer = Trainer(device=device, model=model, train_dataset=train_dataset, valid_dataset=valid_dataset, collate_fn=collate_fn, compute_metrics=compute_metrics, training_config=training_config)
    trainer.train()

if __name__ == '__main__':
    train()