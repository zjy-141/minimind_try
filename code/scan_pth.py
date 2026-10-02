import os
import torch

# ========== 配置区 ==========
ROOT_DIR = r"D:\01_Project\project\2026-project17\minimind\checkpoints" 
OUTPUT_FILE = r"D:\01_Project\project\2026-project17\pth_scan_all.txt"
# ============================

def scan_all_pth(root_dir, output_file):
    # 收集所有 .pth 文件
    pth_files = []
    for dirpath, _, filenames in os.walk(root_dir):
        for f in filenames:
            if f.lower().endswith(".pth"):
                pth_files.append(os.path.join(dirpath, f))

    if not pth_files:
        print("没有找到任何 .pth 文件。")
        return

    with open(output_file, "w", encoding="utf-8") as fout:
        fout.write(f"共找到 {len(pth_files)} 个 .pth 文件\n")
        fout.write("=" * 100 + "\n\n")

        for path in pth_files:
            print(f"正在处理: {path}")
            fout.write("=" * 100 + "\n")
            fout.write(f"文件: {path}\n")
            fout.write(f"大小: {os.path.getsize(path) / 1024 / 1024:.2f} MB\n")

            try:
                state_dict = torch.load(path, map_location="cpu", weights_only=True)
            except Exception as e:
                fout.write(f"⚠️ 加载失败: {e}\n")
                fout.write("尝试用 weights_only=False 加载（仅限可信文件）...\n")
                try:
                    state_dict = torch.load(path, map_location="cpu", weights_only=False)
                except Exception as e2:
                    fout.write(f"❌ 仍然失败: {e2}\n\n")
                    continue

            if not isinstance(state_dict, dict):
                fout.write(f"类型不是字典: {type(state_dict)}，跳过。\n\n")
                continue

            keys = list(state_dict.keys())
            total_params = 0
            for v in state_dict.values():
                if isinstance(v, torch.Tensor):
                    total_params += v.numel()

            fout.write(f"键数量: {len(keys)}\n")
            fout.write(f"总参数量: {total_params:,} (约 {total_params/1e6:.2f}M)\n")

            # 推断层数（针对 HuggingFace 风格）
            layer_indices = set()
            for key in keys:
                if key.startswith("model.layers."):
                    parts = key.split(".")
                    if len(parts) > 2 and parts[2].isdigit():
                        layer_indices.add(int(parts[2]))
            if layer_indices:
                fout.write(f"推断层数: {max(layer_indices) + 1}\n")

            fout.write("\n--- 所有键名与形状 ---\n")
            for key in keys:
                value = state_dict[key]
                if isinstance(value, torch.Tensor):
                    fout.write(f"{key}  ->  shape={tuple(value.shape)}, dtype={value.dtype}\n")
                else:
                    fout.write(f"{key}  ->  {type(value)} (非张量)\n")
            fout.write("\n\n")

    print(f"\n扫描完成，所有数据已保存到: {output_file}")

if __name__ == "__main__":
    scan_all_pth(ROOT_DIR, OUTPUT_FILE)