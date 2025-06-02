import csv
import json
import os


def load_json_file(filepath):
    """加载JSON文件"""
    try:
        with open(filepath, "r", encoding="utf-8") as f:
            return json.load(f)
    except Exception as e:
        print(f"Error loading {filepath}: {e}")
        return None


def find_common_files():
    """找到三个评分员都有的共同文件"""
    json_folder = "/mnt/shareEEx/liuyang/code/emotion_labeling_json/json"
    annotators = ["huangjun", "liuyang", "yuhangbin"]

    # 收集每个评分员的文件列表
    annotator_files = {}
    for annotator in annotators:
        annotator_path = os.path.join(json_folder, annotator)
        if os.path.exists(annotator_path):
            files = [f for f in os.listdir(annotator_path) if f.endswith(".json")]
            annotator_files[annotator] = set(files)
        else:
            print(f"Warning: {annotator_path} does not exist")
            annotator_files[annotator] = set()

    # 找到所有三个人都有的文件
    common_files = annotator_files["huangjun"] & annotator_files["liuyang"] & annotator_files["yuhangbin"]

    print(f"Found {len(common_files)} common files among all three annotators")
    return sorted(common_files)


def analyze_disagreements():
    """分析评分不一致的样本"""
    json_folder = "/mnt/shareEEx/liuyang/code/emotion_labeling_json/json"
    output_folder = "/mnt/shareEEx/liuyang/code/emotion_labeling_json/extracted_data"

    # 确保输出文件夹存在
    os.makedirs(output_folder, exist_ok=True)

    # 三个评分员
    annotators = ["huangjun", "liuyang", "yuhangbin"]

    # 找到共同文件
    common_files = find_common_files()

    if not common_files:
        print("No common files found among all three annotators")
        return

    # 存储不一致的样本
    a_value_disagreements = []
    v_value_disagreements = []

    # 统计计数器
    a_value_count = 0
    v_value_count = 0
    total_samples = 0

    # 处理每个共同文件
    for filename in common_files:
        print(f"Processing {filename}...")

        # 加载三个评分员的数据
        annotator_data = {}
        for annotator in annotators:
            filepath = os.path.join(json_folder, annotator, filename)
            data = load_json_file(filepath)
            if data and isinstance(data, list):
                # 将列表转换为以audio_file为键的字典
                data_dict = {}
                for item in data:
                    if "audio_file" in item:
                        data_dict[item["audio_file"]] = item
                annotator_data[annotator] = data_dict
            else:
                print(f"  Error: Failed to load data for {annotator} in {filename}")
                break

        # 确保三个评分员的数据都加载成功
        if len(annotator_data) != 3:
            print(f"  Skipping {filename} - data loading incomplete")
            continue

        # 找到所有三个评分员都有的wav片段
        common_wav_segments = set(annotator_data["huangjun"].keys())
        for annotator in ["liuyang", "yuhangbin"]:
            common_wav_segments &= set(annotator_data[annotator].keys())

        print(f"  Found {len(common_wav_segments)} common wav segments")

        # 检查每个共同的wav片段
        for wav_segment in common_wav_segments:
            total_samples += 1

            # 提取a_value和v_value
            a_values = []
            v_values = []

            for annotator in annotators:
                item = annotator_data[annotator][wav_segment]
                a_values.append(float(item.get("a_value", 0)))
                v_values.append(float(item.get("v_value", 0)))

            # 检查a_value的差距
            a_max = max(a_values)
            a_min = min(a_values)
            if a_max - a_min > 0.5:
                a_value_count += 1
                a_value_disagreements.append(
                    {
                        "filename": filename,
                        "wav_segment": wav_segment,
                        "huangjun_a_value": a_values[0],
                        "liuyang_a_value": a_values[1],
                        "yuhangbin_a_value": a_values[2],
                        "max_difference": round(a_max - a_min, 2),
                        "username": annotator_data["huangjun"][wav_segment].get("username", ""),
                        "patient_status": annotator_data["huangjun"][wav_segment].get("patient_status", ""),
                        "emotion_type": annotator_data["huangjun"][wav_segment].get("emotion_type", ""),
                    }
                )

            # 检查v_value的差距
            v_max = max(v_values)
            v_min = min(v_values)
            if v_max - v_min > 0.5:
                v_value_count += 1
                v_value_disagreements.append(
                    {
                        "filename": filename,
                        "wav_segment": wav_segment,
                        "huangjun_v_value": v_values[0],
                        "liuyang_v_value": v_values[1],
                        "yuhangbin_v_value": v_values[2],
                        "max_difference": round(v_max - v_min, 2),
                        "username": annotator_data["huangjun"][wav_segment].get("username", ""),
                        "patient_status": annotator_data["huangjun"][wav_segment].get("patient_status", ""),
                        "emotion_type": annotator_data["huangjun"][wav_segment].get("emotion_type", ""),
                    }
                )

    # 保存结果到CSV文件
    print("\n" + "=" * 60)
    print("ANALYSIS RESULTS")
    print("=" * 60)
    print(f"Total processed files: {len(common_files)}")
    print(f"Total processed samples: {total_samples}")
    print(f"A_value disagreements (>0.5): {a_value_count} samples ({a_value_count / total_samples * 100:.2f}%)")
    print(f"V_value disagreements (>0.5): {v_value_count} samples ({v_value_count / total_samples * 100:.2f}%)")

    # 保存a_value不一致的样本
    if a_value_disagreements:
        a_csv_path = os.path.join(output_folder, "a_value_disagreements.csv")
        with open(a_csv_path, "w", newline="", encoding="utf-8") as f:
            fieldnames = ["filename", "wav_segment", "huangjun_a_value", "liuyang_a_value", "yuhangbin_a_value", "max_difference", "username", "patient_status", "emotion_type"]
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(a_value_disagreements)
        print(f"A_value disagreements saved to: {a_csv_path}")
    else:
        print("No A_value disagreements found")

    # 保存v_value不一致的样本
    if v_value_disagreements:
        v_csv_path = os.path.join(output_folder, "v_value_disagreements.csv")
        with open(v_csv_path, "w", newline="", encoding="utf-8") as f:
            fieldnames = ["filename", "wav_segment", "huangjun_v_value", "liuyang_v_value", "yuhangbin_v_value", "max_difference", "username", "patient_status", "emotion_type"]
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(v_value_disagreements)
        print(f"V_value disagreements saved to: {v_csv_path}")
    else:
        print("No V_value disagreements found")

    # 生成详细统计摘要
    summary_path = os.path.join(output_folder, "disagreement_analysis_summary.txt")
    with open(summary_path, "w", encoding="utf-8") as f:
        f.write("情感评分不一致分析摘要报告\n")
        f.write("=" * 60 + "\n\n")
        f.write(f"分析时间: {pd.Timestamp.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n")
        f.write("数据概览:\n")
        f.write(f"  - 共同处理的文件数: {len(common_files)}\n")
        f.write(f"  - 总样本数: {total_samples}\n\n")
        f.write("不一致统计:\n")
        f.write(f"  - A维度评分差距>0.5的样本: {a_value_count} ({a_value_count / total_samples * 100:.2f}%)\n")
        f.write(f"  - V维度评分差距>0.5的样本: {v_value_count} ({v_value_count / total_samples * 100:.2f}%)\n")
        f.write(f"  - 总不一致样本: {a_value_count + v_value_count}\n\n")
        f.write("输出文件:\n")
        f.write("  - a_value_disagreements.csv: A维度评分差距>0.5的详细样本\n")
        f.write("  - v_value_disagreements.csv: V维度评分差距>0.5的详细样本\n\n")
        f.write("共同处理的文件列表:\n")
        for i, filename in enumerate(common_files, 1):
            f.write(f"  {i}. {filename}\n")

    print(f"Detailed summary saved to: {summary_path}")
    print("\nAnalysis completed successfully!")


if __name__ == "__main__":
    try:
        import pandas as pd
    except ImportError:
        print("Installing pandas...")
        os.system("pip install pandas")
        import pandas as pd

    analyze_disagreements()
