import os
import json
import numpy as np

# 获取脚本所在目录的绝对路径
script_dir = os.path.dirname(os.path.abspath(__file__))
# 获取项目根目录路径
project_root = os.path.dirname(script_dir)


def load_annotations():
    """加载所有标注者的JSON数据"""
    base_dir = os.path.join(project_root, "json")
    annotators = ["huangjun", "liuyang", "yuhangbin"]

    # 存储所有标注数据
    all_data = {}
    common_files = set()

    # 首先确定哪些文件是所有标注者共有的
    for annotator in annotators:
        annotator_dir = os.path.join(base_dir, annotator)
        files = [f for f in os.listdir(annotator_dir) if f.endswith("_labels.json")]

        if not common_files:
            common_files = set(files)
        else:
            common_files &= set(files)

    print(f"共同文件: {sorted(common_files)}")

    # 加载共有文件的标注数据
    for filename in common_files:
        file_data = {annotator: [] for annotator in annotators}

        for annotator in annotators:
            file_path = os.path.join(base_dir, annotator, filename)
            with open(file_path, "r") as f:
                try:
                    data = json.load(f)
                    file_data[annotator] = data
                except json.JSONDecodeError:
                    print(f"Error decoding JSON from {file_path}")

        all_data[filename] = file_data

    return all_data, list(annotators)


def find_inconsistent_data(all_data, annotators):
    """
    找出标注不一致的音频数据
    只要V值或A值在三个标注者之间有任何不同就标记为不一致
    """
    inconsistent_data = []
    total_samples = 0

    for filename, file_data in all_data.items():
        print(f"\n处理文件: {filename}")

        # 将列表转换为以audio_file为键的字典
        processed_data = {}
        for annotator in annotators:
            processed_data[annotator] = {}
            for item in file_data[annotator]:
                if isinstance(item, dict) and "audio_file" in item:
                    audio_file = item["audio_file"]
                    processed_data[annotator][audio_file] = item

        # 找出所有标注者共同标注的样本
        common_audio_files = set()
        for annotator in annotators:
            if not common_audio_files:
                common_audio_files = set(processed_data[annotator].keys())
            else:
                common_audio_files &= set(processed_data[annotator].keys())

        print(f"  共同标注的音频文件数: {len(common_audio_files)}")
        total_samples += len(common_audio_files)

        # 检查每个音频样本的一致性
        for audio_file in common_audio_files:
            v_values = []
            a_values = []
            annotations = {}

            # 收集三个标注者的VA值
            for annotator in annotators:
                item = processed_data[annotator][audio_file]
                v_value = item.get("v_value", 0)
                a_value = item.get("a_value", 0)

                v_values.append(v_value)
                a_values.append(a_value)

                annotations[annotator] = {"v_value": v_value, "a_value": a_value, "emotion": item.get("emotion", ""), "audio_file": audio_file}

            # 计算V值和A值的范围
            v_range = max(v_values) - min(v_values)
            a_range = max(a_values) - min(a_values)

            # 判断是否不一致（只要有任何不同就标记为不一致）
            v_inconsistent = v_range > 0
            a_inconsistent = a_range > 0

            if v_inconsistent or a_inconsistent:
                inconsistent_item = {
                    "source_file": filename,
                    "audio_file": audio_file,
                    "inconsistency_type": [],
                    "annotations": annotations,
                }

                if v_inconsistent:
                    inconsistent_item["inconsistency_type"].append("valence")
                if a_inconsistent:
                    inconsistent_item["inconsistency_type"].append("arousal")

                inconsistent_data.append(inconsistent_item)

    print(f"\n总共处理了 {total_samples} 个音频样本")
    return inconsistent_data


def save_inconsistent_data(inconsistent_data):
    """保存不一致的数据到JSON文件"""

    # 创建输出目录
    output_dir = os.path.join(project_root, "extracted_data")
    os.makedirs(output_dir, exist_ok=True)

    # 准备保存的数据
    output_data = {
        "extraction_info": {
            "total_inconsistent_samples": len(inconsistent_data),
            "extraction_criteria": "V值或A值在三个标注者之间有任何不同",
            "annotators": ["huangjun", "liuyang", "yuhangbin"],
        },
        "inconsistent_samples": inconsistent_data,
    }

    # 保存文件
    output_file = os.path.join(output_dir, "inconsistent_va_annotations.json")
    with open(output_file, "w", encoding="utf-8") as f:
        json.dump(output_data, f, ensure_ascii=False, indent=2)

    print(f"\n不一致数据已保存到: {output_file}")
    return output_file


def print_summary(inconsistent_data):
    """打印简单统计摘要"""
    total_inconsistent = len(inconsistent_data)

    v_inconsistent = sum(1 for item in inconsistent_data if "valence" in item["inconsistency_type"])
    a_inconsistent = sum(1 for item in inconsistent_data if "arousal" in item["inconsistency_type"])
    both_inconsistent = sum(1 for item in inconsistent_data if "valence" in item["inconsistency_type"] and "arousal" in item["inconsistency_type"])

    v_only = v_inconsistent - both_inconsistent
    a_only = a_inconsistent - both_inconsistent

    print(f"\n=== 统计结果 ===")
    print(f"V值和A值不一致的样本总数: {total_inconsistent}")
    print(f"  - 仅V值不一致: {v_only} 个")
    print(f"  - 仅A值不一致: {a_only} 个")
    print(f"  - V值和A值都不一致: {both_inconsistent} 个")


def main():
    print("开始提取V值和A值不一致的音频数据...")

    # 1. 加载标注数据
    print("加载标注数据...")
    all_data, annotators = load_annotations()

    # 2. 找出不一致的数据
    print("分析标注一致性...")
    inconsistent_data = find_inconsistent_data(all_data, annotators)

    # 3. 打印统计结果
    print_summary(inconsistent_data)

    # 4. 保存不一致数据到JSON文件
    if inconsistent_data:
        output_file = save_inconsistent_data(inconsistent_data)
        print(f"\n提取完成! 共找到 {len(inconsistent_data)} 个不一致的样本")
    else:
        print("\n所有样本的V值和A值都完全一致!")

    print("分析完成!")


if __name__ == "__main__":
    main()
