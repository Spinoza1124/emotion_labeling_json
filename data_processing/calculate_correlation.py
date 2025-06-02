import os
import json
import numpy as np
from scipy import stats
import pandas as pd

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

    print(f"共同标注的文件: {sorted(common_files)}")

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


def extract_va_values(all_data, annotators):
    """提取VA值用于分析"""
    valence_data = {}
    arousal_data = {}

    for filename, file_data in all_data.items():
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

        # 为每个样本收集所有标注者的VA值
        for audio_file in common_audio_files:
            if audio_file not in valence_data:
                valence_data[audio_file] = {}
                arousal_data[audio_file] = {}

            for annotator in annotators:
                valence_data[audio_file][annotator] = processed_data[annotator][audio_file].get("v_value", 0)
                arousal_data[audio_file][annotator] = processed_data[annotator][audio_file].get("a_value", 0)

    print(f"共处理了 {len(valence_data)} 个音频样本")
    return valence_data, arousal_data


def calculate_correlations(valence_data, arousal_data, annotators):
    """计算标注者之间的相关系数"""

    # 准备数据
    valence_df = pd.DataFrame(valence_data).T  # 转置，行为样本，列为标注者
    arousal_df = pd.DataFrame(arousal_data).T

    print("\n=== V值（Valence）相关性分析 ===")
    print("V值相关系数矩阵:")
    valence_corr = valence_df.corr()
    print(valence_corr.round(3))

    print("\n=== A值（Arousal）相关性分析 ===")
    print("A值相关系数矩阵:")
    arousal_corr = arousal_df.corr()
    print(arousal_corr.round(3))

    # 计算两两之间的显著性检验
    print("\n=== V值显著性检验 (p-values) ===")
    v_pvalues = calculate_pvalues(valence_df, annotators)
    print(v_pvalues.round(4))

    print("\n=== A值显著性检验 (p-values) ===")
    a_pvalues = calculate_pvalues(arousal_df, annotators)
    print(a_pvalues.round(4))

    # 详细的两两相关性报告
    print("\n=== 详细相关性报告 ===")
    for i in range(len(annotators)):
        for j in range(i + 1, len(annotators)):
            ann1, ann2 = annotators[i], annotators[j]

            # V值相关性
            v_corr = valence_corr.iloc[i, j]
            v_pval = v_pvalues.iloc[i, j]

            # A值相关性
            a_corr = arousal_corr.iloc[i, j]
            a_pval = a_pvalues.iloc[i, j]

            print(f"\n{ann1} vs {ann2}:")
            print(f"  V值相关系数: {v_corr:.3f} (p = {v_pval:.4f})")
            print(f"  A值相关系数: {a_corr:.3f} (p = {a_pval:.4f})")

            # 相关性强度解释
            print(f"  V值相关性强度: {interpret_correlation(v_corr)}")
            print(f"  A值相关性强度: {interpret_correlation(a_corr)}")

    return valence_corr, arousal_corr


def calculate_pvalues(df, annotators):
    """计算p值矩阵"""
    n = len(annotators)
    pvalues = np.zeros((n, n))

    for i in range(n):
        for j in range(n):
            if i != j:
                # 检查是否为常量数组
                if np.all(df.iloc[:, i] == df.iloc[0, i]) or np.all(df.iloc[:, j] == df.iloc[0, j]):
                    pvalues[i, j] = np.nan
                else:
                    _, p = stats.pearsonr(df.iloc[:, i], df.iloc[:, j])
                    pvalues[i, j] = p
            else:
                pvalues[i, j] = 0  # 自己与自己的p值设为0

    return pd.DataFrame(pvalues, index=annotators, columns=annotators)


def interpret_correlation(r):
    """解释相关系数强度"""
    r = abs(r)
    if r >= 0.8:
        return "非常强"
    elif r >= 0.6:
        return "强"
    elif r >= 0.4:
        return "中等"
    elif r >= 0.2:
        return "弱"
    else:
        return "非常弱"


def main():
    print("开始计算标注者之间的相关性...")

    # 1. 加载标注数据
    print("加载标注数据...")
    all_data, annotators = load_annotations()

    # 2. 提取VA值
    print("提取VA值...")
    valence_data, arousal_data = extract_va_values(all_data, annotators)

    # 3. 计算相关性
    print("计算相关性...")
    valence_corr, arousal_corr = calculate_correlations(valence_data, arousal_data, annotators)

    print("\n分析完成!")


if __name__ == "__main__":
    main()
