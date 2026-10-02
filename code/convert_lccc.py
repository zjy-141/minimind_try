# 用来处理LCCC数据，使得其变为适合minimind的格式

import json
import os
import re

# ============================================================
# 预编译正则（放在循环外，只编译一次，速度最快）
# 规则：删除所有空格，除非空格两侧都是 ASCII 字母/数字
#      —— 等价于原来 split 循环的逻辑，但用 C 层正则实现
# ============================================================
SPACE_PATTERN = re.compile(r'(?<![A-Za-z0-9])\s+|\s+(?![A-Za-z0-9])')


def clean_text(text):
    """
    清理分词后的文本：
    - 去掉中文/标点之间的空格
    - 保留英文单词、数字之间的空格
    用正则实现，比逐字符 split 循环快 3~5 倍
    """
    text = text.strip()
    if not text:
        return text
    return SPACE_PATTERN.sub('', text)


def convert_lccc_to_minimind(input_path, output_path):
    """将 LCCC-base（整体 JSON 数组）转换为 MiniMind SFT 格式"""
    converted = 0
    skipped = 0

    # 自动创建输出目录
    out_dir = os.path.dirname(output_path)
    if out_dir and not os.path.exists(out_dir):
        os.makedirs(out_dir, exist_ok=True)

    # 一次性读取整个 JSON 文件（格式 B：整体是一个大数组）
    print(f"正在读取 {input_path} ...")
    with open(input_path, 'r', encoding='utf-8') as fin:
        data = json.load(fin)
    print(f"读取完成，共 {len(data)} 条对话，开始转换...")

    with open(output_path, 'w', encoding='utf-8') as fout:
        for dialog in data:
            if not isinstance(dialog, list) or len(dialog) < 2:
                skipped += 1
                continue

            conversations = []
            for i, utterance in enumerate(dialog):
                # 偶数索引 = user，奇数索引 = assistant
                role = "user" if i % 2 == 0 else "assistant"
                content = clean_text(str(utterance))
                if not content:
                    continue
                conversations.append({
                    "role": role,
                    "content": content
                })

            # 确保最后一条是 assistant 的回复（截断多余的 user 发言）
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

    print(f"转换完成：{converted} 条对话，跳过 {skipped} 条")


if __name__ == "__main__":
    # ============ 在这里修改你的输入/输出路径 ============
    INPUT_PATH  = r"D:\01_Project\project\2026-project17\dataset\LCCC-base-split\LCCC-base_valid.json"
    OUTPUT_PATH = r"D:\01_Project\project\2026-project17\dataset\LCCC-base-close\LCCC-base_valid.jsonl"
    # ====================================================

    convert_lccc_to_minimind(INPUT_PATH, OUTPUT_PATH)