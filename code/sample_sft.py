import random

INPUT = r"D:\01_Project\project\2026-project17\minimind\dataset\sft_t2t_mini_full.jsonl"
OUTPUT = r"D:\01_Project\project\2026-project17\minimind\dataset\sft_t2t_mini.jsonl"
TARGET = 50000
TOTAL = 6820506  # 你的 LCCC 全量条数

p = TARGET / TOTAL

count = 0
with open(INPUT, 'r', encoding='utf-8') as fin, open(OUTPUT, 'w', encoding='utf-8') as fout:
    for line in fin:
        if random.random() < p:
            fout.write(line)
            count += 1
            if count >= TARGET:
                break

print(f"采样完成：{count} 条，已写入 {OUTPUT}")