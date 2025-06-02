import json
import os

import numpy as np
from scipy.stats import pearsonr, spearmanr
from sklearn.metrics import cohen_kappa_score, mean_squared_error
from statsmodels.stats.inter_rater import fleiss_kappa


def load_json_data(file_path):
    """加载JSON文件数据"""
    with open(file_path, "r", encoding="utf-8") as f:
        data = json.load(f)
    return data


def main():
    # 三个标注者的目录
    raters = ["huangjun", "liuyang", "yuhangbin"]
    json_dir = "/mnt/shareEEx/liuyang/code/emotion_labeling_json/json"

    # 存储所有标注者的数据
    all_annotations = {}

    # 读取每个标注者的所有JSON文件
    for rater in raters:
        rater_dir = os.path.join(json_dir, rater)
        for json_file in os.listdir(rater_dir):
            if json_file.endswith("_labels.json"):
                file_path = os.path.join(rater_dir, json_file)
                data = load_json_data(file_path)

                # 对于每个音频标注，提取文件名、v_value和a_value
                for item in data:
                    if not isinstance(item, dict) or "audio_file" not in item:
                        continue

                    audio_file = item["audio_file"]
                    v_value = item.get("v_value")
                    a_value = item.get("a_value")

                    # 初始化该音频的标注字典
                    if audio_file not in all_annotations:
                        all_annotations[audio_file] = {"v_values": {}, "a_values": {}}

                    # 保存标注者对该音频的v_value和a_value
                    all_annotations[audio_file]["v_values"][rater] = v_value
                    all_annotations[audio_file]["a_values"][rater] = a_value

    # 找出所有三个标注者都标注过的音频文件
    common_audios = []
    for audio_file, annotations in all_annotations.items():
        if len(annotations["v_values"]) == len(raters) and len(annotations["a_values"]) == len(raters):
            common_audios.append(audio_file)

    print(f"三位标注者共同标注的音频文件数量: {len(common_audios)}")

    # 计算v_value的一致性
    print("\n===== v_value的一致性分析 =====")
    calculate_agreement(all_annotations, common_audios, raters, "v_values", "v_value")

    # 计算a_value的一致性
    print("\n===== a_value的一致性分析 =====")
    calculate_agreement(all_annotations, common_audios, raters, "a_values", "a_value")


def calculate_agreement(all_annotations, common_audios, raters, value_key, display_name):
    """计算多种一致性指标"""
    # 找出所有可能的值
    all_values = set()
    for audio_file in common_audios:
        for rater in raters:
            value = all_annotations[audio_file][value_key].get(rater)
            if value is not None:
                all_values.add(value)

    # 对值进行排序并创建索引
    value_to_idx = {value: idx for idx, value in enumerate(sorted(all_values))}
    n_categories = len(value_to_idx)

    print(f"{display_name}可能的取值: {sorted(all_values)}")

    # 构建Fleiss Kappa的数据矩阵
    table = np.zeros((len(common_audios), n_categories))

    # 填充数据表
    for audio_idx, audio_file in enumerate(common_audios):
        for rater in raters:
            value = all_annotations[audio_file][value_key].get(rater)
            if value is not None:
                table[audio_idx, value_to_idx[value]] += 1

    # 计算Fleiss' Kappa
    kappa = fleiss_kappa(table)

    print(f"1. Fleiss' Kappa值: {kappa:.4f}")
    interpret_kappa(kappa)

    # 创建每对标注者之间的数据
    print("\n2. 每对标注者之间的Cohen's Kappa值:")
    rater_pairs = [(i, j) for i in range(len(raters)) for j in range(i + 1, len(raters))]

    for i, j in rater_pairs:
        rater1, rater2 = raters[i], raters[j]
        ratings1, ratings2 = [], []

        for audio_file in common_audios:
            val1 = all_annotations[audio_file][value_key].get(rater1)
            val2 = all_annotations[audio_file][value_key].get(rater2)
            if val1 is not None and val2 is not None:
                ratings1.append(val1)
                ratings2.append(val2)

        # 处理数据类型问题 - 确保所有值都是字符串类型
        ratings1_str = [str(r) for r in ratings1]
        ratings2_str = [str(r) for r in ratings2]

        # 计算Cohen's Kappa（包括带权重的）
        if len(set(ratings1_str)) > 1 or len(set(ratings2_str)) > 1:  # 避免所有标注完全相同情况
            try:
                # 尝试使用字符串类型计算简单Kappa
                simple_kappa = cohen_kappa_score(ratings1_str, ratings2_str)

                # 对于加权Kappa，需要将值转换为数值
                # 创建唯一值到整数的映射
                unique_values = sorted(set(ratings1 + ratings2))
                value_to_int = {val: idx for idx, val in enumerate(unique_values)}

                # 转换为整数
                ratings1_int = [value_to_int[r] for r in ratings1]
                ratings2_int = [value_to_int[r] for r in ratings2]

                weighted_kappa = cohen_kappa_score(ratings1_int, ratings2_int, weights="linear")
                print(f"{rater1} vs {rater2}: 简单Kappa={simple_kappa:.4f}, 加权Kappa={weighted_kappa:.4f}")

                # 计算相关系数和均方误差 (使用原始数值)
                pearson, _ = pearsonr(ratings1, ratings2)
                spearman, _ = spearmanr(ratings1, ratings2)
                mse = mean_squared_error(ratings1, ratings2)
                print(f"  Pearson相关系数: {pearson:.4f}, Spearman等级相关: {spearman:.4f}, MSE: {mse:.4f}")
            except Exception as e:
                print(f"{rater1} vs {rater2}: 计算Kappa时出错 - {e}")

    # 计算整体百分比一致性
    total_items = len(common_audios)
    exact_matches = 0

    for audio_file in common_audios:
        values = [all_annotations[audio_file][value_key].get(rater) for rater in raters]
        values = [v for v in values if v is not None]
        if len(values) == len(raters) and len(set(values)) == 1:
            exact_matches += 1

    percent_agreement = exact_matches / total_items if total_items > 0 else 0
    print(f"\n3. 整体百分比一致性: {percent_agreement:.2%} ({exact_matches}/{total_items})")

    # 计算每个值的标注一致性
    print(f"\n4. {display_name}各取值的一致性情况:")
    for value in sorted(all_values):
        count = 0
        total = 0
        for audio_file in common_audios:
            values = [all_annotations[audio_file][value_key].get(rater) for rater in raters]
            values = [v for v in values if v is not None]
            if value in values:
                total += 1
                if len(values) == len(raters) and values.count(value) == len(raters):
                    count += 1
        if total > 0:
            agreement = count / total
            print(f"值 {value}: {agreement:.2%}一致 ({count}/{total})")


def interpret_kappa(kappa):
    """解释Kappa值"""
    if kappa < 0:
        print("解释: 差于随机标注")
    elif kappa < 0.2:
        print("解释: 轻微一致")
    elif kappa < 0.4:
        print("解释: 一般一致")
    elif kappa < 0.6:
        print("解释: 中度一致")
    elif kappa < 0.8:
        print("解释: 显著一致")
    else:
        print("解释: 几乎完全一致")


if __name__ == "__main__":
    main()
