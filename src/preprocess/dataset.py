"""
Dataset 工具模块

提供从本地 HuggingFace DatasetDict 创建 PyTorch DataLoader 的辅助函数，
方便训练脚本直接复用，避免重复样板代码。
"""
import logging
from typing import Dict, Optional

from datasets import DatasetDict
from torch.utils.data import DataLoader

logger = logging.getLogger(__name__)


def to_dataloader(
    dataset_dict: DatasetDict,
    split: str = 'train',
    batch_size: int = 32,
    shuffle: Optional[bool] = None,
) -> DataLoader:
    """
    将 DatasetDict 的某个 split 转换为 PyTorch DataLoader。

    Args:
        dataset_dict: 已分词完成的 DatasetDict
        split: 要加载的 split 名称（如 'train' / 'valid' / 'test'）
        batch_size: 批大小
        shuffle: 是否打乱；默认训练集打乱，其余不打乱

    Returns:
        torch.utils.data.DataLoader
    """
    if shuffle is None:
        shuffle = (split == 'train')

    dataset = dataset_dict[split]
    dataset.set_format(type='torch')

    dataloader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
    )
    logger.info('已为 split=%s 创建 DataLoader，共 %d 个样本', split, len(dataset))
    return dataloader


def get_label_mapping(dataset_dict: DatasetDict) -> Dict[int, str]:
    """
    从 DatasetDict 中提取 id 到 label 的映射，便于推理阶段还原预测结果。

    Args:
        dataset_dict: 包含 ClassLabel 列的 DatasetDict

    Returns:
        dict: {0: '类别0', 1: '类别1', ...}
    """
    label_feature = dataset_dict['train'].features['labels']
    return {idx: name for idx, name in enumerate(label_feature.names)}
