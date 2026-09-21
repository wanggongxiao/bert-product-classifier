# bert-product-classifier

基于 BERT 的中文商品标题分类项目。目前仓库包含原始数据预处理、Hugging Face Dataset 转换，以及基于 PyTorch 的训练、验证、混合精度、早停和检查点逻辑。

> 项目仍在开发中：数据预处理流程可以独立运行；训练入口与配置模块之间的接口尚未完全对齐，暂时不能保证端到端训练命令可直接执行。

## 项目结构

```text
bert-product-classifier/
|-- README.md
|-- src/
|   |-- configuration/
|   |   `-- config.py       # 项目路径和预训练模型配置
|   |-- preprocess/
|   |   |-- process.py      # TSV 清洗、标签编码、分词与持久化
|   |   `-- dataset.py      # DatasetDict 转 DataLoader 工具
|   `-- runner/
|       `-- train.py        # 训练、验证、早停与检查点逻辑
|-- data/
|   |-- raw/                # 原始 train/valid/test 数据（需自行创建）
|   `-- processed/          # 预处理后的 DatasetDict 和 labels.json
|-- pretrained/             # 本地预训练模型目录
|-- models/                 # 模型检查点输出目录
`-- logs/                   # TensorBoard 日志目录
```

`data/`、`models/`、`logs/` 等运行时目录不会提交到 Git，需要时由使用者创建或由脚本生成。

## 环境准备

建议使用 Python 3.9 或更高版本，并在虚拟环境中安装依赖：

```bash
python -m venv .venv
```

Windows PowerShell：

```powershell
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
pip install datasets transformers torch scikit-learn tensorboard tqdm
```

Linux/macOS：

```bash
source .venv/bin/activate
python -m pip install --upgrade pip
pip install datasets transformers torch scikit-learn tensorboard tqdm
```

首次运行预处理或训练时，Transformers 会从 Hugging Face 下载 `bert-base-chinese`，因此需要能够访问模型仓库；也可以在配置中改为本地模型路径。

## 数据格式

在 `data/raw/` 下准备三个 UTF-8 编码的 TSV 文件：

```text
data/raw/train.txt
data/raw/valid.txt
data/raw/test.txt
```

每个文件第一行为表头，字段名必须是 `label` 和 `text_a`，字段之间使用制表符分隔：

```text
label\ttext_a
服装\t男士纯棉短袖T恤
数码\t苹果 iPhone 手机壳
食品\t原味混合坚果
```

三个数据集应使用一致的类别标签。

## 数据预处理

在项目根目录执行：

```bash
python -m src.preprocess.process
```

预处理流程会：

1. 加载 `train`、`valid` 和 `test` 三个 TSV 文件；
2. 过滤标签或商品标题为空的样本；
3. 将字符串标签编码为 `ClassLabel`；
4. 使用中文 BERT tokenizer 对 `text_a` 分词；
5. 将 DatasetDict 保存到 `data/processed/`，并生成 `labels.json`。

## 训练模块

`src/runner/train.py` 中的 `TrainingConfig` 提供以下训练参数：

| 参数 | 默认值 | 说明 |
| --- | --- | --- |
| `epochs` | `10` | 训练轮数 |
| `batch_size` | `16` | 批大小 |
| `learning_rate` | `5e-5` | 学习率 |
| `output_dir` | `./models` | 检查点目录 |
| `log_dir` | `./logs` | TensorBoard 日志目录 |
| `save_steps` | `100` | 检查点保存间隔 |
| `early_stop_metric` | `loss` | 早停指标 |
| `early_stop_patience` | `3` | 早停容忍次数 |
| `use_amp` | `True` | 是否启用自动混合精度 |

训练器目前包含以下逻辑：

- Adam 优化器和动态 padding；
- CUDA 自动混合精度训练；
- 验证集 loss、accuracy 和 weighted F1；
- TensorBoard loss 记录；
- 检查点保存与恢复；
- 早停状态管理。

训练入口仍需完成配置字段、数据集加载方式及早停方法的对接，完成前不建议直接执行 `src/runner/train.py`。

## 配置

路径配置集中在 `src/configuration/config.py`：

| 配置项 | 说明 |
| --- | --- |
| `ROOT_PATH` | 项目根目录 |
| `RAW_DATA_DIR` | 原始数据目录 |
| `PROCESSED_DATA_DIR` | 预处理数据目录 |
| `LOG_DIR` | TensorBoard 日志目录 |
| `MODELS_DIR` | 模型输出目录 |
| `PRE_TRAINED_DIR` | 本地预训练模型目录 |
| `PRE_TRAINED_MODEL_NAME` | 默认预训练模型名称 |

## 当前进度

- [x] 项目路径和基础配置
- [x] TSV 数据加载、清洗与标签编码
- [x] BERT tokenizer 预处理与 DatasetDict 持久化
- [x] Dataset/DataLoader 辅助工具
- [x] 训练、验证、AMP 和检查点逻辑
- [ ] 打通可直接运行的端到端训练入口
- [ ] 增加独立测试集评估与推理脚本
- [ ] 增加 RESTful API 服务
- [ ] 增加自动化测试和 CI

## TensorBoard

训练入口打通并产生日志后，可在项目根目录查看训练曲线：

```bash
tensorboard --logdir logs
```

浏览器访问终端输出的地址（通常为 `http://localhost:6006`）。
