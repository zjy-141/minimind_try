# 查看pth文件内容

import torch
import numpy as np

PATH = r"D:\01_Project\project\2026-project17\minimind\checkpoints\pretrain_768.pth"
OUTPUT = r"D:\01_Project\project\2026-project17\pth_inspect.txt"

state_dict = torch.load(PATH, map_location="cpu", weights_only=True)

with open(OUTPUT, "w", encoding="utf-8") as f:
    def w(s=""):
        print(s)
        f.write(s + "\n")

    def stats(name, tensor):
        # 统一用 numpy 处理
        t = tensor.float().numpy().flatten()
        w(f"\n{'='*80}")
        w(f"张量: {name}")
        w(f"  形状: {tuple(tensor.shape)}")
        w(f"  dtype: {tensor.dtype}")
        w(f"  元素总数: {t.size}")
        w(f"  最小值: {t.min():.6f}")
        w(f"  最大值: {t.max():.6f}")
        w(f"  均值: {t.mean():.6f}")
        w(f"  标准差: {t.std():.6f}")
        w(f"  绝对值均值: {np.abs(t).mean():.6f}")
        w(f"  零值占比: {(t == 0).mean()*100:.2f}%")
        w(f"  NaN 数量: {np.isnan(t).sum()}")
        w(f"  Inf 数量: {np.isinf(t).sum()}")

    # ============ 1. 小张量完整数值 ============
    w("\n" + "#"*80)
    w("# 第一部分：小张量完整数值")
    w("#"*80)

    for name in ["model.norm.weight",
                 "model.layers.0.self_attn.q_norm.weight",
                 "model.layers.0.self_attn.k_norm.weight",
                 "model.layers.0.input_layernorm.weight",
                 "model.layers.0.post_attention_layernorm.weight"]:
        if name in state_dict:
            w(f"\n--- {name} (完整) ---")
            w(str(state_dict[name].float().numpy()))

    # ============ 2. 大张量统计 ============
    w("\n" + "#"*80)
    w("# 第二部分：关键大张量的统计信息")
    w("#"*80)

    for name in ["model.embed_tokens.weight",
                 "model.layers.0.self_attn.q_proj.weight",
                 "model.layers.0.self_attn.k_proj.weight",
                 "model.layers.0.self_attn.v_proj.weight",
                 "model.layers.0.self_attn.o_proj.weight",
                 "model.layers.0.mlp.gate_proj.weight",
                 "model.layers.0.mlp.up_proj.weight",
                 "model.layers.0.mlp.down_proj.weight",
                 "lm_head.weight"]:
        if name in state_dict:
            stats(name, state_dict[name])

    # ============ 3. 大张量切片 ============
    w("\n" + "#"*80)
    w("# 第三部分：大张量切片预览（前 3x8 个数值）")
    w("#"*80)

    for name in ["model.embed_tokens.weight",
                 "model.layers.0.self_attn.q_proj.weight",
                 "model.layers.0.mlp.gate_proj.weight",
                 "lm_head.weight"]:
        if name in state_dict:
            w(f"\n--- {name} [0:3, 0:8] ---")
            w(str(state_dict[name][0:3, 0:8].float().numpy()))

    # ============ 4. 词嵌入分析 ============
    w("\n" + "#"*80)
    w("# 第四部分：词嵌入层分析（是否有全零行？）")
    w("#"*80)

    emb = state_dict["model.embed_tokens.weight"].float()
    row_norms = emb.norm(dim=1)
    w(f"\n词嵌入矩阵形状: {tuple(emb.shape)}")
    w(f"每一行的 L2 范数:")
    w(f"  最小值: {row_norms.min():.6f}  (第 {row_norms.argmin().item()} 行)")
    w(f"  最大值: {row_norms.max():.6f}  (第 {row_norms.argmax().item()} 行)")
    w(f"  均值: {row_norms.mean():.6f}")
    w(f"  全零行数量: {(row_norms == 0).sum().item()}")

    # ============ 5. 权重绑定检查 ============
    w("\n" + "#"*80)
    w("# 第五部分：lm_head 与 embed_tokens 是否为同一份权重")
    w("#"*80)

    emb = state_dict["model.embed_tokens.weight"]
    head = state_dict["lm_head.weight"]
    w(f"\nembed_tokens 形状: {tuple(emb.shape)}")
    w(f"lm_head     形状: {tuple(head.shape)}")
    w(f"两者数值是否完全相同: {torch.equal(emb, head)}")
    if not torch.equal(emb, head):
        diff = (emb - head).abs().max().item()
        w(f"最大差异: {diff:.8f}")

print(f"\n详细数据已保存到: {OUTPUT}")