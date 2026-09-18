# bert-product-classifier

一个基于 BERT 的中文商品标题分类系统，支持数据预处理、模型训练、评估和 RESTful API 推理。

---

## 项目目标

利用预训练的 `bert-base-chinese` 模型，对中文商品标题进行多分类（例如：服饰 / 数码 / 食品 等）。
整个流程包含四个阶段：

1. **数据预处理** — 读取原始数据、清洗、划分标签、分词、保存为 HuggingFace Dataset
2. **模型训练** — 基于 BERT 微调分类模型
3. **模型评估** — 在测试集上评估准确率、精确率、召回率等指标
4. **API 推理** — 提供 RESTful 接口供线上调用

---

## 项目结构

```
bert-product-classifier/
├── README.md                  # 项目说明（本文件）
├── src/
│   ├── configuration/
│   │   └── config.py          # 全局路径与配置
│   └── preprocess/
│       └── process.py         # 数据预处理主流程
├── data/
│   ├── raw/                   # 原始数据（train.txt / valid.txt / test.txt）
│   └── processed/             # 预处理后的数据 + labels.json
├── pretrained/                # 预训练模型存放目录（如 bert-base-chinese）
├── models/                    # 微调后的模型保存目录
└── logs/                      # 训练日志目录
```

---

## 数据格式

原始数据采用 **TSV** 格式（两列：`label\ttext_a`），分为三个文件：

- `data/raw/train.txt`  — 训练集
- `data/raw/valid.txt`  — 验证集
- `data/raw/test.txt`   — 测试集

示例：

```
服饰	男士纯棉短袖T恤
数码	苹果iPhone 15 Pro手机壳
食品	原味坚果混合装
```

---

## 环境依赖

```bash
pip install datasets transformers torch scikit-learn fastapi uvicorn
```

---

## 使用方法

### 1. 数据预处理

将 `train.txt` / `valid.txt` / `test.txt` 放入 `data/raw/` 目录，然后运行：

```bash
python -m src.preprocess.process
```

该脚本会完成以下工作：

- 加载原始 CSV/TXT 数据
- 过滤空值
- 构建标签集合并转换为 `ClassLabel`
- 保存 `labels.json`
- 使用 `bert-base-chinese` 分词器对文本进行编码
- 将处理后的数据集保存到 `data/processed/`

### 2. 模型训练

（待实现：`src/train/train.py`）

### 3. 模型评估

（待实现：`src/evaluate/evaluate.py`）

### 4. API 推理

（待实现：`src/api/app.py`）

启动方式（计划）：

```bash
uvicorn src.api.app:app --host 0.0.0.0 --port 8000
```

调用示例：

```bash
curl -X POST http://localhost:8000/predict \
     -H "Content-Type: application/json" \
     -d '{"text": "男士纯棉短袖T恤"}'
```

---

## 关键路径配置

所有路径都集中在 `src/configuration/config.py`：

| 变量 | 含义 |
|---|---|
| `ROOT_PATH` | 项目根目录 |
| `RAW_DATA_DIR` | 原始数据目录 `data/raw/` |
| `PROCESSED_DATA_DIR` | 预处理后数据目录 `data/processed/` |
| `LOG_DIR` | 训练日志目录 `logs/` |
| `MODELS_DIR` | 模型保存目录 `models/` |
| `PRE_TRAINED_DIR` | 预训练模型目录 `pretrained/` |
| `PRE_TRAINED_MODEL_NAME` | 预训练模型名称，默认 `bert-base-chinese` |

---

## 模块说明

### `src/configuration/config.py`
集中管理所有路径常量，避免散落在代码各处。使用 `pathlib.Path`，跨平台兼容。

### `src/preprocess/process.py`
数据预处理主流程，按职责拆分为 6 个小函数：
- `load_raw_dataset()` — 加载原始 TSV
- `clean_dataset()` — 过滤空值
- `build_labels()` — 构建标签集合并转为 `ClassLabel`
- `save_labels()` — 保存 `labels.json`
- `tokenize_dataset()` — 分词 + 添加 `labels` 字段
- `save_dataset()` — 保存为 HuggingFace Dataset 格式

每个步骤都通过 `logger` 输出进度，错误可追溯。

### `src/preprocess/dataset.py`
提供 `to_dataloader()` 与 `get_label_mapping()` 两个工具函数，供训练 / 推理脚本复用，避免重复样板代码。

---

## 当前进度

- [x] 项目结构搭建
- [x] 全局配置（`config.py`）
- [x] 数据预处理脚本（`process.py`）
- [x] Dataset 工具函数（`dataset.py`）
- [ ] 模型训练脚本
- [ ] 模型评估脚本
- [ ] RESTful API 服务

---

## 后续改进方向

1. **修复 `process.py` 中已知的代码 bug**（例如 `ClassLabel` 拼写、`sorted` 函数、`RAW_DATA_DIR` 等）
2. 增加训练 / 评估 / 推理模块
3. 增加单元测试与 CI
4. 编写 Docker 镜像，支持一键部署
5. 增加模型版本管理与推理性能监控
