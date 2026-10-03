# MiniMind 本地训练与部署 README

> 在 Windows 10/11 + RTX 5060 Laptop（8GB 显存 **sm_120**）上跑通 MiniMind 的实操记录：预训练用 MiniMind 官方数据集，中文 SFT 用 **LCCC-base**，最后转 **GGUF** 部署到 **WSL 里的 Ollama**。
> 本文件是"整体路线 + 关键命令"的简明版；逐步排查细节、参数取舍与踩坑经过见 [操作指南.md](操作指南.md)（v6.0）。
> 环境已在本机复测：`python 3.10.21`、`torch 2.9.1+cu128`（CUDA build 12.8）、`cuda_available True`、`capability (12, 0)`、`NVIDIA GeForce RTX 5060 Laptop GPU`。

> **当前进度（2026-10-02 核对）**：环境 ✅ ｜ 预训练 ✅（loss 7.69 → 1.49）｜ 第一次 SFT ⚠️（用默认 lr=1e-5、19.98 万条、2 epoch，loss 停在 2.6，模型不会对话）｜ 5 万条子集 ✅（49,939 条）｜ 修正后 SFT ⚠️（权重已出但**没留日志**）｜ GGUF ⚠️（`gguf/minimind1.gguf` 早于新权重，**需按 §九 重做**）。

---

## 一、硬件与系统环境

| 项目 | 值 |
|---|---|
| GPU | NVIDIA GeForce RTX 5060 Laptop，8GB 显存 |
| 架构 | sm_120（Blackwell），计算能力 (12, 0) |
| 驱动 | 592.01，支持 CUDA 12.9 |
| CPU | Intel Ultra 9 275HX，24 核 / 24 线程 |
| 内存 | 31.4 GB |
| 操作系统 | Windows 10/11；Ollama 装在 WSL 中 |

> **8GB 显存是唯一硬瓶颈**：训练固定 `batch_size=8` + 梯度累积，并且必须同时看**功耗**判断是否真的在算（见 §八）。

---

## 二、软件栈

| 组件 | 版本 / 路径 |
|---|---|
| Anaconda | `D:\05_DevTools\Programme\anaconda` |
| Conda 环境 | `minimind`（`envs\minimind`） |
| Python | 3.10.21（实测） |
| PyTorch | 2.9.1+cu128（稳定版，支持 sm_120；实测 CUDA build 12.8） |
| 项目代码 | `minimind\`（MiniMind 上游代码） |
| 预训练数据 | 官方 `pretrain_t2t_mini.jsonl`（1,270,238 条） |
| SFT 数据 | LCCC-base 转换后抽样 5 万条 → `minimind\dataset\sft_t2t_mini.jsonl` |
| GGUF 转换 | `llama.cpp\`（新版结构，转换逻辑在 `conversion\base.py`） |
| 部署 | WSL + Ollama + GGUF |

---

## 三、快速开始

```cmd
:: 1. 打开 Anaconda Prompt（PowerShell 里 conda activate 常无反应）
:: 2. 进入项目目录
cd /d D:\01_Project\project\2026-project17

:: 3. 激活环境
conda activate minimind

:: 4. 验证 GPU 与矩阵运算
python code\torch_test.py
```

预期输出（末行数值每次都不同）：

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
├── 操作指南.md                      # v6.0，细节与踩坑以它为准
├── README.md                        # 本文件
├── .gitignore                       # 数据/权重/日志/gguf 不入库（只跟踪代码与文档）
├── code/                            # 自写脚本（旧文档误写成项目根目录）
│   ├── convert_lccc.py              # LCCC JSON → MiniMind SFT 格式（无 max_samples）
│   ├── sample_sft.py                # 从全量随机抽 5 万条（TARGET=50000）
│   ├── torch_test.py                # GPU 验证（旧文档称 test_gpu.py，不存在）
│   └── check.py / scan_pth.py / inspect_pth.py    # 权重排查，输出在 log\pth_*.txt
├── dataset/                         # 原始 LCCC 数据（6.24GB）
│   ├── LCCC-base-split/*.json       # 解压后的原始数据（train / valid / test）
│   ├── LCCC-base-close/*.jsonl      # 转换结果
│   ├── pretrain_t2t_mini.jsonl      # 预训练数据副本
│   ├── sft_t2t_mini.jsonl           # 全量转换副本
│   └── delete/sft_t2t_mini.jsonl    # 无脚本引用的遗留文件，可删
├── gguf/minimind1.gguf              # 已转换的 GGUF（122.2MB）
├── llama.cpp/                       # 转换工具（无 .git 的解压副本）
├── log/                             # 训练/转换日志（train_*.txt、llama.cpp.txt、pth_*.txt）
└── minimind/                        # MiniMind 代码本体
    ├── model/                       # 结构代码；配置类为 model_minimind.py 的 MiniMindConfig
    ├── trainer/                     # 训练脚本 train_*.py
    ├── scripts/                     # convert_model.py；Modelfile（FROM 已失效，勿用）
    ├── dataset/                     # 见 §六
    ├── out/                         # pretrain_768.pth、full_sft_768.pth
    ├── checkpoints/                 # 同名权重 + 649MB 的 *_resume.pth（断点续训）
    ├── minimind-3/                  # HF 导出目录（脚本硬编码，不是 minimind-hf）
    └── requirements.txt
```

---

## 五、环境安装（已装好可跳过）

### 5.1 Conda 环境

```cmd
conda create -n minimind python=3.10 -y
conda activate minimind
```

### 5.2 安装 PyTorch 稳定版 cu128

> 不要用 Nightly。RTX 5060（sm_120）实测 `2.9.1+cu128` 稳定可用。

```cmd
pip uninstall torch torchvision torchaudio -y
pip install torch==2.9.1 torchvision==0.24.1 torchaudio==2.9.1 --index-url https://download.pytorch.org/whl/cu128
```

验证：

```python
import torch
print(torch.__version__, torch.version.cuda)
print(torch.cuda.is_available(), torch.cuda.get_device_name(0), torch.cuda.get_device_capability(0))
x = torch.randn(1000, 1000, device="cuda"); print(torch.mm(x, x).sum().item())
```

### 5.3 安装项目依赖

> 先打开 `minimind\requirements.txt`，**删掉或注释 `torch`、`torchvision`、`torchaudio` 三行**，避免覆盖刚装好的 cu128。

```cmd
cd /d D:\01_Project\project\2026-project17\minimind
pip install -r requirements.txt -i https://mirrors.aliyun.com/pypi/simple
```

### 5.4 验证 flash-attn 导入

```cmd
cd /d D:\01_Project\project\2026-project17\minimind
python -c "import sys; sys.path.insert(0,'.'); import model.model_minimind; print('import OK')"
```

报 `flash_attn` 错误时，在 `minimind\model\model_minimind.py` 的 `MiniMindConfig` 里设 `flash_attn=False` 走手动注意力。**当前版本没有 `LMConfig.py`**，旧文档的路径会找不到文件。

---

## 六、数据准备

### 6.1 预训练数据（官方）

LCCC-base 是对话数据，不用于预训练：

```cmd
cd /d D:\01_Project\project\2026-project17\minimind\dataset
wget https://www.modelscope.cn/datasets/gongjy/minimind_dataset/resolve/master/pretrain_t2t_mini.jsonl
```

### 6.2 LCCC-base 转成 MiniMind SFT 格式

本工作区的 LCCC 数据是**已解压的整体 JSON 数组**（`dataset\LCCC-base-split\*.json`），用 `code\convert_lccc.py` 转换：`open()` 直接读整个文件（**不走 gzip**），并且**没有 `max_samples` 参数**，会全量转换。

```cmd
cd /d D:\01_Project\project\2026-project17
python code\convert_lccc.py
```

格式对照（LCCC 一行 = 一个对话的字符串数组；MiniMind 一行 = 一个 `conversations` 对象）：

```json
["今天天气真好", "是啊，适合出去玩"]
```

```json
{"conversations": [{"role": "user", "content": "今天天气真好"}, {"role": "assistant", "content": "是啊，适合出去玩"}]}
```

### 6.3 抽 5 万条子集（SFT 实际使用的数据）

```cmd
cd /d D:\01_Project\project\2026-project17
python code\sample_sft.py
```

按 `TARGET=50000` 从 `minimind\dataset\sft_t2t_mini_full.jsonl`（全量 6,820,506 条）随机抽样，直接输出到 SFT 脚本默认读取的 `minimind\dataset\sft_t2t_mini.jsonl`。当前是 **49,939 条**：源文件读完时没攒够 5 万（正常统计波动，标准差约 220），不影响使用。

最终目录：

```text
minimind/dataset/
├── pretrain_t2t_mini.jsonl        1.24GB  预训练（1,270,238 条）
├── sft_t2t_mini.jsonl             10MB    5 万条子集（SFT 默认读取）
└── sft_t2t_mini_full.jsonl        1.39GB  全量转换结果（抽样源）
```

---

## 七、训练

### 7.1 预训练

```cmd
cd /d D:\01_Project\project\2026-project17\minimind\trainer
python train_pretrain.py --max_seq_len 512 --batch_size 8 --accumulation_steps 4 > ..\..\log\train_pretrain_run2.txt 2>&1
```

- 本机已跑完一轮：2 个 epoch，loss 7.69 → **1.49**，权重在 `out\pretrain_768.pth`。
- 显存不足就降到 `--batch_size 4 --accumulation_steps 8`。
- `log\train_pretrain.txt` 是首轮记录，重跑请写到新文件名，别覆盖。

### 7.2 SFT（关键：学习率必须调高）

第一次 SFT 用了脚本默认参数（起始 lr **1e-5**，余弦衰减到 1e-6）、19.98 万条、2 个 epoch，loss 停在 2.6 左右 —— 日志末尾的 `lr: 0.00000100` 是**衰减终点**，不是设定值。修正后的命令：

```cmd
cd /d D:\01_Project\project\2026-project17\minimind\trainer
python train_full_sft.py --learning_rate 5e-5 --epochs 5 --batch_size 8 --accumulation_steps 4 --data_path ../dataset/sft_t2t_mini.jsonl > ..\..\log\train_full_sft_2nd.txt 2>&1
```

- 观察 loss 应降到 **1.5 左右**；权重写入 `out\full_sft_768.pth`（`checkpoints\` 下同名文件与 `*_resume.pth`）。
- **务必重定向到 `log\`**：上一轮修正后的训练没留日志，`5e-5 / 5 epoch` 是否生效已无法复核。
- `log\train_full_sft.txt` 是**未修正**那次的记录。

### 7.3 LoRA（可选）

脚本默认读 `../dataset/lora_medical.jsonl`，该文件当前不存在，需先复制一份：

```cmd
cd /d D:\01_Project\project\2026-project17\minimind\dataset
copy sft_t2t_mini.jsonl lora_medical.jsonl

cd ..\trainer
python train_lora.py
```

若输出的 `Trainable Params` 等于总参数量，说明原模型没冻结好，需检查脚本。

---

## 八、训练监控

```cmd
nvidia-smi -l 2
```

| 指标 | 健康 | 显存溢出 / 卡住 |
|---|---|---|
| GPU 利用率 | 97~99% | 100%（看起来一样） |
| 功耗 | 71~93 W | 35~43 W |
| 温度 | 持续升至 76°C+ | 60°C 且不升 |
| 显存占用 | < 7.5 GB | 接近或达到 8 GB |

> **判据**：100% 利用率 + 低功耗 + 温度不升 = 在等 PCIe，不是在算数。显存溢出会伪装成 100% 利用率，务必看功耗。
> 可视化加 `--use_wandb` 即可：MiniMind 内部用 SwanLab，接口与参数名兼容 WandB。

---

## 九、转 GGUF 并部署到 WSL Ollama

### 9.1 导出 HuggingFace 格式

```cmd
cd /d D:\01_Project\project\2026-project17\minimind\scripts
python convert_model.py --format transformers --input_dir ../out --output_dir ../minimind-3
```

> `--output_dir` 传什么都无效：脚本第 133 行把 `transformers_path` 硬编码为 `../minimind-3`，实际导出目录就是 `minimind\minimind-3`（架构为 `Qwen3ForCausalLM`）。

### 9.2 转 GGUF

> Qwen2 pre-tokenizer 哈希补丁**已存在**于 `llama.cpp\conversion\base.py`（第 1939 行，`res = "qwen2"`），无需再改；旧文档让改 `convert_hf_to_gguf.py`，在新版里那个函数已经不在该文件。

```cmd
cd /d D:\01_Project\project\2026-project17\minimind\scripts
python D:\01_Project\project\2026-project17\llama.cpp\convert_hf_to_gguf.py ../minimind-3 --outfile minimind.gguf
```

生成约 122MB 的 `minimind.gguf`（本工作区已移到 `gguf\minimind1.gguf`，转换日志见 `log\llama.cpp.txt`）。

> **重训 SFT 后必须重做 9.1 + 9.2**，否则部署的还是旧权重导出的模型。

### 9.3 在 WSL 里注册 Ollama 模型

```bash
mkdir -p ~/minimind-ollama
cp /mnt/d/01_Project/project/2026-project17/gguf/minimind1.gguf ~/minimind-ollama/
cd ~/minimind-ollama
nano Modelfile
```

```dockerfile
FROM ./minimind1.gguf
SYSTEM "你的名字叫MiniMind，你是一个乐于助人、知识渊博的AI助手。"
PARAMETER repeat_penalty 1
PARAMETER stop "<|im_start|>"
PARAMETER stop "<|im_end|>"
PARAMETER temperature 0.2
```

```bash
ollama create minimind-lccc -f ./Modelfile
ollama run minimind-lccc
```

- 也可以直接写 `FROM /mnt/d/01_Project/project/2026-project17/gguf/minimind1.gguf`；报 `/mnt/d` 权限错误就按上面复制到家目录。
- **不要用**仓库里的 `minimind\scripts\Modelfile`：它写的是 `FROM ./minimind.gguf`（该文件不存在），`temperature` 还是 0.7。
- Ollama 会在 `http://localhost:11434` 提供兼容 OpenAI 的 API。

---

## 十、常见问题

| 问题 | 解决方案 |
|---|---|
| `conda activate` 在 PowerShell 无反应 | 改用 Anaconda Prompt |
| CMD 跨盘符切换失败 | 用 `cd /d D:\...` |
| `pip install` 报 WinError 5 | `taskkill /f /im python.exe`，或用 `--user` |
| `%TEMP%` ACL 拒绝访问 | 把 TEMP 换到有完全控制权的目录 |
| `github.com` 超时 | 用 ModelScope / hf-mirror |
| `CUDA out of memory` | 降 `batch_size`、增 `accumulation_steps` |
| 训练"看起来在跑但很慢" | 看功耗，低于 50W 说明在等 PCIe |
| `flash_attn` 导入失败 | `model\model_minimind.py` 的 `MiniMindConfig` 设 `flash_attn=False` |
| PyTorch 不支持 sm_120 | 用 cu128 稳定版，别用 Nightly |
| SFT loss 不降（>2.5） | lr 提到 `5e-5`、数据减到 5 万条、跑 5 个 epoch |
| GGUF 转换报 BPE 错误 | 确认 `conversion\base.py` 里有 Qwen2 哈希映射 |
| 找不到 `convert_lccc.py` / `LMConfig.py` / `minimind/scripts/minimind.gguf` | 实际位置是 `code\`、`model\model_minimind.py` 的 `MiniMindConfig`、`gguf\minimind1.gguf` |
| 按旧文档用 `max_samples` 转换报错 | 脚本没这个参数，5 万条子集用 `code\sample_sft.py` 抽样 |
| SFT 参数事后查不到 | 当次忘了重定向日志；重跑时加 `> ..\..\log\train_full_sft_2nd.txt 2>&1` |

---

## 十一、参考

- MiniMind GitHub：`https://github.com/jingyaogong/minimind`
- MiniMind 数据集：`https://www.modelscope.cn/datasets/gongjy/minimind_dataset`
- LCCC-base：HuggingFace / ModelScope 搜索 `lccc base`
- PyTorch cu128 安装：`https://download.pytorch.org/whl/cu128`
- llama.cpp：`https://github.com/ggml-org/llama.cpp`
- Ollama：`https://ollama.com`

---

## 十二、版本记录

| 版本 | 说明 |
|---|---|
| v1.0 | 初始环境搭建 |
| v2.0 | 修正 Anaconda Prompt、PyTorch 稳定版、batch_size 安全值 |
| v3.0 | 支持直接从 GitHub 下载项目代码 |
| v4.0 | 接入 LCCC-base 中文 SFT 数据 |
| **本 README** | 按工作区实际核对后重写（Windows + WSL 版）：修正脚本路径、配置类名、HF 导出目录、GGUF 位置与 Modelfile；补上 5 万条抽样流程与"必须留训练日志"；环境本机复测（Python 3.10.21 / torch 2.9.1+cu128 / (12, 0)）。细节见 [操作指南.md](操作指南.md) v6.0 |
