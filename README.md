# bert-product-classifier

基于 `bert-base-chinese` 的中文商品标题分类项目，包含数据预处理、模型训练、评估和 RESTful API 推理模块。

## 项目结构

```text
bert-product-classifier/
├── README.md
├── src/
│   ├── configuration/config.py  # 路径和模型配置
│   ├── preprocess/process.py    # 原始 TSV 数据预处理
│   ├── preprocess/dataset.py    # Dataset 和 DataLoader 工具
│   └── runner/train.py          # 训练配置和训练器
├── data/
│   ├── raw/                     # train.txt、valid.txt、test.txt
│   └── processed/               # 处理后的 Dataset 和 labels.json
├── pretrained/                  # 预训练模型目录
├── models/                      # 微调模型输出目录
└── logs/                        # TensorBoard 日志目录
```

## 数据格式

原始数据使用 TSV 格式，每行包含 `label\ttext_a`：

```text
服装\t男士纯棉短袖T恤
数码\t苹果 iPhone 15 Pro 手机壳
食品\t原味坚果混合装
```

数据文件放在 `data/raw/` 下：`train.txt`、`valid.txt` 和 `test.txt`。

## 安装依赖

```bash
pip install datasets transformers torch scikit-learn tensorboard tqdm
```

## 使用方法

### 数据预处理

```bash
python -m src.preprocess.process
```

该命令会读取原始 TSV，清洗空值，建立 `ClassLabel`，使用 BERT tokenizer 编码，并将结果保存到 `data/processed/`。

### 模型训练

训练配置和训练器位于 `src/runner/train.py`。可通过 `TrainingConfig` 设置训练超参数：

```python
from src.runner.train import TrainingConfig

training_config = TrainingConfig(
    epochs=10,
    batch_size=16,
    learning_rate=5e-5,
    output_dir='./models',
    log_dir='./logs',
    save_steps=100,
    early_stop_metric='loss',
    early_stop_patience=3,
    use_amp=True,
)
```

当前 `train` 类已完成设备、模型、数据集、优化器、TensorBoard 和 AMP 等训练组件的初始化；完整的训练循环、验证、保存和早停逻辑仍在完善中。

### 模型评估和 API 推理

评估模块和 API 模块尚未完成。计划使用以下命令启动 API：

```bash
uvicorn src.api.app:app --host 0.0.0.0 --port 8000
```

## 路径配置

统一配置位于 `src/configuration/config.py`：

| 配置项 | 说明 |
|---|---|
| `ROOT_PATH` | 项目根目录 |
| `RAW_DATA_DIR` | 原始数据目录 |
| `PROCESSED_DATA_DIR` | 处理后数据目录 |
| `LOG_DIR` | 训练日志目录 |
| `MODELS_DIR` | 模型输出目录 |
| `PRE_TRAINED_DIR` | 预训练模型目录 |
| `PRE_TRAINED_MODEL_NAME` | 默认值为 `bert-base-chinese` |

## 当前进度

- [x] 项目结构和全局配置
- [x] 数据预处理脚本
- [x] Dataset/DataLoader 工具
- [x] 训练配置和训练器初始化框架（`src/runner/train.py`）
- [ ] 完整模型训练循环
- [ ] 模型评估脚本
- [ ] RESTful API 服务

## 后续计划

1. 完成训练循环、验证、断点保存和早停。
2. 增加准确率、F1 等评估指标和测试脚本。
3. 增加 RESTful API、单元测试和 CI。
