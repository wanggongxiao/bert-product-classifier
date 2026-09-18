"""
数据预处理主流程

负责将原始 TSV 数据转换为 BERT 友好的 HuggingFace Dataset，并保存到本地。

输入：data/raw/{train,valid,test}.txt，TSV 格式（label\\ttext_a）
输出：data/processed/ 下的 DatasetDict 与 labels.json
"""
import logging
import sys
from pathlib import Path

# 确保 src/ 目录在导入路径中，使 `from configuration import config` 可用
sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent / 'src'))

from datasets import ClassLabel, load_dataset
from transformers import AutoTokenizer

from configuration import config

# 配置日志，方便追踪预处理进度与错误
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def load_raw_dataset():
    """
    从 data/raw/ 加载 train/valid/test 三个 TSV 文件。

    Returns:
        datasets.DatasetDict: 包含 train/valid/test 三个 split
    """
    logger.info('开始加载原始数据...')
    dataset_dict = load_dataset(
        'csv',
        data_files={
            'train': str(config.RAW_DATA_DIR / 'train.txt'),
            'valid': str(config.RAW_DATA_DIR / 'valid.txt'),
            'test':  str(config.RAW_DATA_DIR / 'test.txt'),
        },
        delimiter='\t',
    )
    logger.info('原始数据加载完成：%s', dataset_dict)
    return dataset_dict


def clean_dataset(dataset_dict):
    """
    过滤掉 label 或 text_a 为空的样本。

    Args:
        dataset_dict: 原始 DatasetDict

    Returns:
        datasets.DatasetDict: 清洗后的 DatasetDict
    """
    logger.info('开始清洗数据（过滤空值）...')

    def is_valid(example):
        return example['label'] is not None and example['text_a'] is not None

    dataset_dict = dataset_dict.filter(is_valid)
    logger.info('数据清洗完成')
    return dataset_dict


def build_labels(dataset_dict):
    """
    从训练集中提取全部 label，排序后转为 ClassLabel 类型。

    同时把所有 split 中的 label 列转换为 ClassLabel。

    Args:
        dataset_dict: 清洗后的 DatasetDict

    Returns:
        tuple: (dataset_dict, sorted_labels)
    """
    logger.info('开始构建标签集合...')
    # 训练集上出现的所有 label，去重并排序，保证映射稳定
    all_labels = sorted(set(dataset_dict['train']['label']))
    logger.info('共发现 %d 个类别：%s', len(all_labels), all_labels)

    # 将所有 split 的 label 列转换为 ClassLabel，便于后续模型训练
    dataset_dict = dataset_dict.cast_column('label', ClassLabel(names=all_labels))
    return dataset_dict, all_labels


def save_labels(labels, save_path: Path):
    """
    将标签列表保存为 JSON 文件。

    Args:
        labels: 排序后的标签列表
        save_path: 保存路径
    """
    import json
    save_path.parent.mkdir(parents=True, exist_ok=True)
    with open(save_path, 'w', encoding='utf-8') as f:
        json.dump(labels, f, ensure_ascii=False, indent=2)
    logger.info('标签已保存至 %s', save_path)


def tokenize_dataset(dataset_dict):
    """
    使用 bert-base-chinese 分词器对 text_a 进行编码，并把 label 复制到 labels 字段。

    Args:
        dataset_dict: 已转换 label 列类型的 DatasetDict

    Returns:
        datasets.DatasetDict: 分词后的 DatasetDict
    """
    logger.info('加载分词器：%s', config.PRE_TRAINED_MODEL_NAME)
    tokenizer = AutoTokenizer.from_pretrained(
        config.PRE_TRAINED_DIR / config.PRE_TRAINED_MODEL_NAME
    )

    def tokenize(batch):
        # 对 text_a 字段进行分词；truncation=True 防止超出最大长度
        inputs = tokenizer(batch['text_a'], truncation=True)
        # BERT 分类任务约定使用 labels 字段作为分类标签
        inputs['labels'] = batch['label']
        return inputs

    logger.info('开始批量分词...')
    dataset_dict = dataset_dict.map(
        tokenize,
        batched=True,
        remove_columns=['text_a', 'label'],
    )
    logger.info('分词完成。训练集样本示例：%s', dataset_dict['train'])
    return dataset_dict


def save_dataset(dataset_dict, save_path: Path):
    """
    将处理后的 DatasetDict 保存到本地。

    Args:
        dataset_dict: 处理后的 DatasetDict
        save_path: 保存路径
    """
    save_path.mkdir(parents=True, exist_ok=True)
    dataset_dict.save_to_disk(str(save_path))
    logger.info('数据集已保存至 %s', save_path)


def process():
    """数据预处理主入口：依次执行 加载 -> 清洗 -> 标签构建 -> 分词 -> 保存。"""
    # 1. 加载原始数据
    dataset_dict = load_raw_dataset()

    # 2. 数据清洗
    dataset_dict = clean_dataset(dataset_dict)

    # 3. 构建标签集合 & 转换列类型
    dataset_dict, all_labels = build_labels(dataset_dict)

    # 4. 保存标签文件
    save_labels(all_labels, config.PROCESSED_DATA_DIR / 'labels.json')

    # 5. 分词
    dataset_dict = tokenize_dataset(dataset_dict)

    # 6. 保存数据集
    save_dataset(dataset_dict, config.PROCESSED_DATA_DIR)

    logger.info('数据预处理全部完成 ✅')


if __name__ == '__main__':
    process()
