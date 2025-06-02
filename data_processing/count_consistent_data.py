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

    # 根据文件夹结构，三个人共同的文件是spk2-1-1到spk2-5-1
    common_files = ["spk2-1-1_labels.json", "spk2-2-1_labels.json", "spk2-3-1_labels.json", "spk2-4-1_labels.json", "spk2-5-1_labels.json"]

    print(f"处理共同标注的文件: {common_files}")

    # 加载共有文件的标注数据
    for filename in common_files:
        file_data = {annotator: [] for annotator in annotators}

        for annotator in annotators:
            file_path = os.path.join(base_dir, annotator, filename)
            if os.path.exists(file_path):
                with open(file_path, "r") as f:
                    try:
                        data = json.load(f)
                        file_data[annotator] = data
                    except json.JSONDecodeError:
                        print(f"Error decoding JSON from {file_path}")
            else:
                print(f"File not found: {file_path}")

        all_data[filename] = file_data

    return all_data, list(annotators)


def count_consistent_data(all_data, annotators):
    """
    统计标注一致的音频数据
    V值和A值在三个标注者之间完全相同的样本
    """
    consistent_data = []
    total_samples = 0
    v_consistent_count = 0
    a_consistent_count = 0
    both_consistent_count = 0

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

            # 判断是否一致（所有值都相同）
            v_consistent = len(set(v_values)) == 1  # 只有一个唯一值说明完全一致
            a_consistent = len(set(a_values)) == 1

            # 统计各种一致性情况
            if v_consistent:
                v_consistent_count += 1
            if a_consistent:
                a_consistent_count += 1
            if v_consistent and a_consistent:
                both_consistent_count += 1

                # 保存完全一致的样本信息
                consistent_item = {
                    "source_file": filename,
                    "audio_file": audio_file,
                    "v_value": v_values[0],  # 所有值都相同，取第一个
                    "a_value": a_values[0],
                    "annotations": annotations,
                }
                consistent_data.append(consistent_item)

    print(f"\n总共处理了 {total_samples} 个音频样本")

    return {
        "total_samples": total_samples,
        "v_consistent_count": v_consistent_count,
        "a_consistent_count": a_consistent_count,
        "both_consistent_count": both_consistent_count,
        "v_only_consistent": v_consistent_count - both_consistent_count,
        "a_only_consistent": a_consistent_count - both_consistent_count,
        "consistent_data": consistent_data,
    }


def save_consistent_data(results):
    """保存一致的数据到JSON文件"""

    # 创建输出目录
    output_dir = os.path.join(project_root, "extracted_data")
    os.makedirs(output_dir, exist_ok=True)

    # 准备保存的数据
    output_data = {
        "statistics": {
            "total_samples": results["total_samples"],
            "v_consistent_count": results["v_consistent_count"],
            "a_consistent_count": results["a_consistent_count"],
            "both_consistent_count": results["both_consistent_count"],
            "v_only_consistent": results["v_only_consistent"],
            "a_only_consistent": results["a_only_consistent"],
            "consistency_rate": {
                "v_consistency_rate": results["v_consistent_count"] / results["total_samples"] * 100,
                "a_consistency_rate": results["a_consistent_count"] / results["total_samples"] * 100,
                "both_consistency_rate": results["both_consistent_count"] / results["total_samples"] * 100,
            },
        },
        "extraction_info": {
            "extraction_criteria": "V值和A值在三个标注者之间完全相同",
            "annotators": ["huangjun", "liuyang", "yuhangbin"],
        },
        "consistent_samples": results["consistent_data"],
    }

    # 保存文件
    output_file = os.path.join(output_dir, "consistent_va_annotations.json")
    with open(output_file, "w", encoding="utf-8") as f:
        json.dump(output_data, f, ensure_ascii=False, indent=2)

    print(f"\n一致数据已保存到: {output_file}")
    return output_file


def print_consistency_summary(results):
    """打印一致性统计摘要"""
    total = results["total_samples"]
    v_consistent = results["v_consistent_count"]
    a_consistent = results["a_consistent_count"]
    both_consistent = results["both_consistent_count"]
    v_only = results["v_only_consistent"]
    a_only = results["a_only_consistent"]

    print(f"\n=== 一致性统计结果 ===")
    print(f"总样本数: {total}")
    print(f"V值完全一致的样本: {v_consistent} 个 ({v_consistent / total * 100:.1f}%)")
    print(f"A值完全一致的样本: {a_consistent} 个 ({a_consistent / total * 100:.1f}%)")
    print(f"V值和A值都完全一致的样本: {both_consistent} 个 ({both_consistent / total * 100:.1f}%)")
    print(f"")
    print(f"细分统计:")
    print(f"  - 仅V值一致: {v_only} 个 ({v_only / total * 100:.1f}%)")
    print(f"  - 仅A值一致: {a_only} 个 ({a_only / total * 100:.1f}%)")
    print(f"  - V值和A值都一致: {both_consistent} 个 ({both_consistent / total * 100:.1f}%)")

    # 计算不一致的样本
    inconsistent = total - both_consistent
    print(f"  - 存在不一致: {inconsistent} 个 ({inconsistent / total * 100:.1f}%)")


def main():
    print("开始统计V值和A值一致的音频数据...")

    # 1. 加载标注数据
    print("加载标注数据...")
    all_data, annotators = load_annotations()

    # 2. 统计一致的数据
    print("分析标注一致性...")
    results = count_consistent_data(all_data, annotators)

    # 3. 打印统计结果
    print_consistency_summary(results)

    # 4. 保存一致数据到JSON文件
    if results["consistent_data"]:
        output_file = save_consistent_data(results)
        print(f"\n统计完成! 共找到 {results['both_consistent_count']} 个完全一致的样本")
    else:
        print("\n没有找到完全一致的样本!")

    print("分析完成!")


if __name__ == "__main__":
    main()
