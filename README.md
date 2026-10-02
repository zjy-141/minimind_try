# MiniMind 本地训练与部署 README

> ⚠️ **本文档已过期（v4.0 时期编写），请以 [操作指南.md](操作指南.md)（v6.0）为唯一事实源。**
> 下列内容与当前工作区不符，正文对应位置已就地标注：`convert_lccc.py` 实际在 `code/`（项目根与 `minimind/` 下都没有）；`test_gpu.py` 不存在，实际是 `code/torch_test.py`；`LMConfig.py` 已不存在（配置类为 `minimind/model/model_minimind.py` 的 `MiniMindConfig`）；HF 导出目录被脚本硬编码为 `minimind/minimind-3`（不是 `minimind-hf`）；GGUF 的 Qwen2 哈希补丁在 `llama.cpp/conversion/base.py` 且**已应用**；LCCC 数据是已解压的 `.json`（不是 `.gz`）；GGUF 实际位于 `gguf/minimind1.gguf`。

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

:: 4. 验证 GPU（实际脚本是 code\torch_test.py，没有 test_gpu.py）
python code\torch_test.py
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
├── 操作指南.md                      # v6.0，唯一事实源（本 README 已过期）
├── code/                            # 实际脚本都在这里
│   ├── convert_lccc.py              # LCCC-base → MiniMind SFT 格式（无 max_samples）
│   ├── sample_sft.py                # 从全量抽 5 万条子集
│   ├── torch_test.py                # GPU 验证脚本
│   └── check.py / scan_pth.py / inspect_pth.py
├── dataset/                         # 原始 LCCC 数据（6.24GB）
│   ├── pretrain_t2t_mini.jsonl
│   ├── sft_t2t_mini.jsonl
│   ├── LCCC-base-split/*.json       # 解压后的原始数据（不是 .gz）
│   ├── LCCC-base-close/*.jsonl      # 转换结果
│   └── delete/sft_t2t_mini.jsonl
├── gguf/minimind1.gguf              # 即 §9.1 生成的 minimind.gguf（改名移位后）
├── llama.cpp/                       # 无 .git 的解压副本
├── log/                             # 训练与转换日志
├── minimind/                        # MiniMind 项目代码
│   ├── model/                       # 模型结构（配置类为 model_minimind.py 的 MiniMindConfig）
│   ├── trainer/                     # 训练脚本
│   ├── scripts/                     # convert_model.py、Modelfile（FROM 已失效，勿用）
│   ├── dataset/                     # pretrain_t2t_mini.jsonl / sft_t2t_mini.jsonl / sft_t2t_mini_full.jsonl
│   ├── out/                         # pretrain_768.pth、full_sft_768.pth
│   ├── checkpoints/                 # 含 649MB 的 *_resume.pth（旧文档未提）
│   ├── minimind-3/                  # HF 导出目录（脚本硬编码，非 minimind-hf）
│   ├── requirements.txt
│   └── README.md
└── README.md                        # 本文件（已过期）
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

若报错，在 `minimind/model/model_minimind.py` 的 `MiniMindConfig` 中设置 `flash_attn=False`，使用手动注意力实现。（当前版本已没有 `LMConfig.py`，旧路径会找不到文件。）

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

> **本工作区的数据已经解压**：`dataset/LCCC-base-split/*.json`（整体 JSON 数组），因此下面基于 `gzip.open` 的脚本**不适用于现有数据**，请改用 `code/convert_lccc.py`（见 [操作指南.md](操作指南.md) §2.2.2）。

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

转换脚本：实际是 **`code\convert_lccc.py`**（旧版 README 把它写在项目根目录，`minimind\` 下也没有该文件）。它用 `open()` 读取整体 JSON 数组，**不是**逐行 `gzip` 流式读取，而且**没有 `max_samples` 参数**：

```python
def convert_lccc_to_minimind(input_path, output_path):   # 注意：没有 max_samples
    ...
```

运行：

```cmd
cd /d D:\01_Project\project\2026-project17
python code\convert_lccc.py
```

**再截取 5 万条子集**（SFT 实际使用的数据，旧版 README 缺这一步）：

```cmd
python code\sample_sft.py
```

它按 `TARGET=50000` 随机抽样，直接输出到 `minimind\dataset\sft_t2t_mini.jsonl`（即 SFT 脚本默认读取的文件名），无需再 `copy`。

最终 `minimind/dataset/` 目录（3 个文件；项目根另有 `dataset/` 原始数据目录，见 [操作指南.md](操作指南.md) §2.2.3）：

```text
minimind/dataset/
├── pretrain_t2t_mini.jsonl        1.24GB  预训练
├── sft_t2t_mini.jsonl             10MB    5 万条子集（现有 49,939 条）
└── sft_t2t_mini_full.jsonl        1.39GB  全量转换结果（抽样源）
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
python convert_model.py --format transformers --input_dir ../out --output_dir ../minimind-3
```

> `--output_dir` 传什么都无效：`convert_model.py` 第 133 行把 `transformers_path` 硬编码为 `../minimind-3`，实际导出目录就是 `minimind/minimind-3`（旧版 README 写的 `../minimind-hf` 并不存在）。

使用 llama.cpp 转换（要用绝对路径，`llama.cpp` 不在 `minimind/scripts` 下）：

```cmd
python D:\01_Project\project\2026-project17\llama.cpp\convert_hf_to_gguf.py ../minimind-3 --outfile minimind.gguf
```

Qwen2 pre-tokenizer 哈希补丁**已经应用**在新版 llama.cpp 的 `llama.cpp/conversion/base.py` 第 1939 行（该函数在新版本中已从 `convert_hf_to_gguf.py` 移到这里，所以旧版 README 让改 `convert_hf_to_gguf.py` 会找不到函数）：

```python
if chkhsh == "a39372c92460bbbac950b7bc25ddc61768b9aa10ae4bd07d3336ef66f6b9fd41":
    # ref: MiniMind (Qwen2 tokenizer)
    res = "qwen2"
```

生成的 GGUF 在本工作区位于 `gguf/minimind1.gguf`（122.2MB），`minimind/scripts/` 下没有 gguf 文件。

### 9.2 加载到 Ollama

本工作区的 Ollama 装在 **WSL** 里（Windows 侧没有 `ollama` 命令），做法以 [操作指南.md](操作指南.md) §6.5 为准。要点：

- GGUF 路径为 `/mnt/d/01_Project/project/2026-project17/gguf/minimind1.gguf`；
- 在 WSL 家目录新建 Modelfile（如 `~/Modelfile_minimind`）；
- `temperature` 用 **0.2**（本 README 旧版写的 0.7 与指南不一致）；
- 仓库里的 `minimind/scripts/Modelfile` 写的是 `FROM ./minimind.gguf`，该文件不存在，**不要直接用**。

```dockerfile
FROM /mnt/d/01_Project/project/2026-project17/gguf/minimind1.gguf
SYSTEM "你的名字叫MiniMind，你是一个乐于助人、知识渊博的AI助手。"
PARAMETER repeat_penalty 1
PARAMETER stop "<|im_start|>"
PARAMETER stop "<|im_end|>"
PARAMETER temperature 0.2
```

```bash
ollama create minimind-lccc -f ~/Modelfile_minimind
ollama run minimind-lccc
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
| `flash_attn` 导入失败 | 在 `minimind/model/model_minimind.py` 的 `MiniMindConfig` 中设 `flash_attn=False`（当前版本没有 `LMConfig.py`） |
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
| **本 README** | 基于操作指南 v4.0 生成，适用于 RTX 5060 Laptop / 8GB 显存（**已过期**：请以 [操作指南.md](操作指南.md) v6.0 为准） |