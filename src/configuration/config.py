"""
全局配置模块

集中管理项目的所有路径常量，方便统一修改。
使用 pathlib.Path，支持跨平台路径操作。
"""
from pathlib import Path

# 项目根目录 = 当前文件所在目录的上三级（src/configuration/config.py -> 项目根）
ROOT_PATH = Path(__file__).resolve().parent.parent.parent

# 各子目录路径
RAW_DATA_DIR = ROOT_PATH / 'data' / 'raw'              # 原始数据目录
PROCESSED_DATA_DIR = ROOT_PATH / 'data' / 'processed'  # 预处理后数据目录
LOG_DIR = ROOT_PATH / 'logs'                           # 训练日志目录
MODELS_DIR = ROOT_PATH / 'models'                      # 模型保存目录
PRE_TRAINED_DIR = ROOT_PATH / 'pretrained'             # 预训练模型目录

# 预训练模型名称（用于 AutoTokenizer / AutoModel.from_pretrained）
PRE_TRAINED_MODEL_NAME = 'bert-base-chinese'
model_name = 'google-bert/bert-base-chinese'
train_file = RAW_DATA_DIR / 'train.txt'
valid_file = RAW_DATA_DIR / 'dev.txt'
eporchs = 20