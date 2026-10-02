# MiniMind 本地训练与部署 README

> 在 Windows + RTX 5060 Laptop（8GB 显存）上从零搭建 MiniMind 的完整实操记录。  
> 使用 **LCCC-base** 作为中文 SFT 语料，预训练使用 MiniMind 官方数据集。  
> 当前环境已通过 GPU 运算验证：`CUDA 可用`、`计算能力 (12, 0)`、`GPU 运算测试通过`。

---

## 一、硬件与系统环境

| 项目 | 值 |
|---|---|
| GPU | NVIDIA GeForce RTX 5060 Laptop，8GB 显存 |
| 架构 | sm_120（Blackwell） |
| 驱动 | 592.01，CUDA 12.9 |
| CPU | Intel Ultra 9 275HX，24 核 / 24 线程 |
| 内存 | 31.4 GB |
| D 盘可用 | 112.9 GB / 551.6 GB |
| 操作系统 | Windows 10/11 |

> **显存约束**：8GB 是训练时的唯一硬瓶颈。必须使用 `batch_size=8` + 梯度累积，并时刻关注功耗，避免显存溢出伪装成正常训练。

---

## 二、软件栈

| 组件 | 版本 / 路径 |
|---|---|
| Anaconda | `D:\05_DevTools\Programme\anaconda` |
| Conda 环境 | `minimind` |
| Python | 3.10.21 |
| PyTorch | 2.9.1+cu128（稳定版，已实测支持 sm_120） |
| 项目代码 | MiniMind（GitHub 下载） |
| 预训练数据 | `pretrain_t2t_mini.jsonl` |
| SFT 数据 | LCCC-base 转换后的 `sft_t2t_mini.jsonl` |
| 部署 | Ollama + GGUF |

---

## 三、快速开始

```cmd
:: 1. 打开 Anaconda Prompt
:: 2. 进入项目目录
cd /d D:\01_Project\project\2026-project17

:: 3. 激活环境
conda activate minimind

:: 4. 验证 GPU
python test_gpu.py
```

预期输出：

```text
CUDA 是否可用: True
GPU 名称: NVIDIA GeForce RTX 5060 Laptop GPU
计算能力: (12, 0)
GPU 运算测试通过: 19369.453125
```

---

## 四、目录结构

```text
2026-project17/
├── minimind/                        # MiniMind 项目代码
│   ├── model/                       # 模型结构代码
│   ├── trainer/                     # 训练脚本
│   ├── scripts/                     # 辅助脚本
│   ├── dataset/                     # 数据集目录
│   │   ├── pretrain_t2t_mini.jsonl  # 预训练数据
│   │   ├── sft_t2t_mini.jsonl       # SFT 数据（由 LCCC-base 转换）
│   │   └── lora_identity.jsonl      # 可选 LoRA 数据
│   ├── requirements.txt
│   ├── LMConfig.py
│   └── README.md
├── convert_lccc.py                  # LCCC-base 转换脚本
├── test_gpu.py                      # GPU 验证脚本
└── README.md                        # 本文件
```

---

## 五、环境安装

### 5.1 创建 Conda 环境

```cmd
conda create -n minimind python=3.10 -y
conda activate minimind
```

### 5.2 安装 PyTorch（稳定版 cu128）

> 不要使用 Nightly。RTX 5060 实测稳定版 `2.9.1+cu128` 完全可用。

```cmd
pip uninstall torch torchvision torchaudio -y
pip install torch==2.9.1 torchvision==0.24.1 torchaudio==2.9.1 --index-url https://download.pytorch.org/whl/cu128
```

验证：

```python
import torch
print(torch.cuda.is_available())
print(torch.cuda.get_device_name(0))
print(torch.cuda.get_device_capability(0))
x = torch.randn(1000, 1000, device="cuda")
y = torch.mm(x, x)
print(y.sum().item())
```

### 5.3 安装项目依赖

> 先打开 `requirements.txt`，删除或注释掉 `torch`、`torchvision`、`torchaudio` 三行，避免覆盖已装好的 PyTorch。

```cmd
pip install -r requirements.txt -i https://mirrors.aliyun.com/pypi/simple
```

### 5.4 验证 flash-attn

```python
import model.model_minimind as m
print("import OK")
```

若报错，在 `LMConfig.py` 中设置 `flash_attn=False`，使用手动注意力实现。

---

## 六、数据准备

### 6.1 预训练数据（MiniMind 官方）

LCCC-base 是对话数据，不用于预训练。预训练继续使用 MiniMind 官方数据集：

```cmd
cd /d D:\01_Project\project\2026-project17\minimind
mkdir dataset
cd dataset

wget https://www.modelscope.cn/datasets/gongjy/minimind_dataset/resolve/master/pretrain_t2t_mini.jsonl
```

### 6.2 LCCC-base 下载

```cmd
wget https://hf-mirror.com/datasets/silver/lccc/resolve/main/lccc_base_train.jsonl.gz
```

或从 HuggingFace / ModelScope 手动下载。

### 6.3 LCCC-base 格式转换

LCCC-base 原始格式：

```json
["今天天气真好", "是啊，适合出去玩", "你有什么推荐的地方吗", "公园或者郊外都不错"]
```

MiniMind SFT 格式：

```json
{"conversations": [
  {"role": "user", "content": "今天天气真好"},
  {"role": "assistant", "content": "是啊，适合出去玩"},
  {"role": "user", "content": "你有什么推荐的地方吗"},
  {"role": "assistant", "content": "公园或者郊外都不错"}
]}
```

转换脚本 `convert_lccc.py`：

```python
import json
import gzip

def convert_lccc_to_minimind(input_path, output_path, max_samples=None):
    converted = 0
    skipped = 0

    with gzip.open(input_path, 'rt', encoding='utf-8') as fin, \
         open(output_path, 'w', encoding='utf-8') as fout:

        for line in fin:
            line = line.strip()
            if not line:
                continue

            dialog = json.loads(line)

            if len(dialog) < 2:
                skipped += 1
                continue

            conversations = []
            for i, utterance in enumerate(dialog):
                role = "user" if i % 2 == 0 else "assistant"
                conversations.append({
                    "role": role,
                    "content": utterance.strip()
                })

            while conversations and conversations[-1]["role"] != "assistant":
                conversations.pop()

            if len(conversations) < 2:
                skipped += 1
                continue

            fout.write(json.dumps(
                {"conversations": conversations},
                ensure_ascii=False
            ) + "\n")
            converted += 1

            if max_samples and converted >= max_samples:
                break

    print(f"转换完成：{converted} 条对话，跳过 {skipped} 条")

if __name__ == "__main__":
    convert_lccc_to_minimind(
        "lccc_base_train.jsonl.gz",
        "lccc_sft_minimind.jsonl",
        max_samples=500000
    )
```

运行：

```cmd
python convert_lccc.py
```

将生成的 `lccc_sft_minimind.jsonl` 复制为 MiniMind 默认 SFT 数据名：

```cmd
copy lccc_sft_minimind.jsonl minimind\dataset\sft_t2t_mini.jsonl
```

最终 `dataset/` 目录：

```text
dataset/
├── pretrain_t2t_mini.jsonl
└── sft_t2t_mini.jsonl
```

---

## 七、训练

### 7.1 预训练

```cmd
cd /d D:\01_Project\project\2026-project17\minimind\trainer
python train_pretrain.py --max_seq_len 512 --batch_size 8 --accumulation_steps 4
```

- `batch_size=8` 是 8GB 显存的安全起点。
- `accumulation_steps=4` 等效 batch_size = 32。
- 若显存不足，降到 `--batch_size 4 --accumulation_steps 8`。

### 7.2 SFT 监督微调

```cmd
python train_full_sft.py --max_seq_len 512 --batch_size 8 --accumulation_steps 4
```

LCCC-base 对话较短，显存占用通常低于官方 SFT 数据。若显存充裕，可尝试 `--batch_size 12`。

### 7.3 LoRA 微调

```cmd
python train_lora.py
```

LoRA 只训练极少量参数，是 8GB 显存下做领域适配的最佳选择。

---

## 八、训练监控

训练时必须同时关注 **GPU 利用率** 和 **功耗**。仅看利用率会被欺骗。

```cmd
nvidia-smi -l 2
```

| 指标 | 健康 | 溢出 / 卡住 |
|---|---|---|
| GPU 利用率 | 97~99% | 100%（看起来一样） |
| 功耗 | 71~93 W | 35~43 W |
| 温度 | 持续升至 76°C+ | 60°C 且不升 |
| 显存占用 | < 7.5 GB | 接近或达到 8 GB |

> **判据**：100% 利用率 + 低功耗 + 温度不升 = 在等 PCIe，不是在算数。  
> 显存溢出会伪装成 100% 利用率，务必看功耗。

可视化：

```cmd
python train_pretrain.py --use_wandb --batch_size 8 --accumulation_steps 4
```

MiniMind 内部使用 SwanLab，接口兼容 WandB，参数名不变。

---

## 九、部署到 Ollama

### 9.1 转换为 GGUF

```cmd
cd /d D:\01_Project\project\2026-project17\minimind\scripts
python convert_model.py --format transformers --input_dir ../out --output_dir ../minimind-hf
```

使用 llama.cpp 转换：

```cmd
python llama.cpp/convert_hf_to_gguf.py ./minimind-hf --outfile minimind.gguf
```

在 `convert_hf_to_gguf.py` 的 `get_vocab_base_pre` 函数末尾添加：

```python
if res is None:
    res = "qwen2"
```

### 9.2 加载到 Ollama

新建 `minimind.modelfile`：

```text
FROM /path/to/minimind.gguf
SYSTEM "你的名字叫MiniMind，你是一个乐于助人、知识渊博的AI助手。"
PARAMETER repeat_penalty 1
PARAMETER stop "<|im_start|>"
PARAMETER stop "<|im_end|>"
PARAMETER temperature 0.7
```

```cmd
ollama create minimind-local -f minimind.modelfile
ollama run minimind-local
```

Ollama 自动在 `http://localhost:11434` 提供兼容 OpenAI 的 API。

---

## 十、常见问题

| 问题 | 解决方案 |
|---|---|
| `conda activate` 在 PowerShell 无反应 | 改用 Anaconda Prompt |
| CMD 跨盘符切换失败 | 使用 `cd /d D:\...` |
| `pip install` 报 WinError 5 | 先 `taskkill /f /im python.exe`，再用 `--user` 模式 |
| `%TEMP%` ACL 拒绝访问 | 更换 TEMP 到有完全控制权的目录 |
| `github.com` 超时 | 使用 ModelScope 或 hf-mirror |
| `CUDA out of memory` | 降 `batch_size`，增 `accumulation_steps` |
| 训练“看起来在跑但很慢” | 检查功耗，低于 50W 说明在等 PCIe |
| `flash_attn` 导入失败 | 在 `LMConfig.py` 中设 `flash_attn=False` |
| PyTorch 不支持 sm_120 | 使用 cu128 稳定版，不要用 Nightly |

---

## 十一、参考

- MiniMind GitHub：`https://github.com/jingyaogong/minimind`
- MiniMind 数据集：`https://www.modelscope.cn/datasets/gongjy/minimind_dataset`
- LCCC-base：HuggingFace / ModelScope 搜索 `lccc base`
- PyTorch cu128 安装：`https://download.pytorch.org/whl/cu128`
- Ollama：`https://ollama.com`

---

## 十二、版本记录

| 版本 | 说明 |
|---|---|
| v1.0 | 初始环境搭建 |
| v2.0 | 修正 Anaconda Prompt、PyTorch 稳定版、batch_size 安全值 |
| v3.0 | 支持直接从 GitHub 下载项目代码 |
| v4.0 | 接入 LCCC-base 中文 SFT 数据，增加格式转换与注意事项 |
| **本 README** | 基于操作指南 v4.0 生成，适用于 RTX 5060 Laptop / 8GB 显存 |