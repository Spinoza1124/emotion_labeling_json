import os
import json
import numpy as np
import pandas as pd
import pingouin as pg

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
                # JSON中VA值存储在"v_value"和"a_value"字段中
                valence_data[audio_file][annotator] = processed_data[annotator][audio_file].get("v_value", 0)
                arousal_data[audio_file][annotator] = processed_data[annotator][audio_file].get("a_value", 0)

    # 转换为DataFrame格式，行为样本，列为标注者
    valence_df = pd.DataFrame(valence_data).T
    arousal_df = pd.DataFrame(arousal_data).T

    print(f"共处理了 {len(valence_df)} 个音频样本")

    # 添加详细统计分析
    print(f"\n=== 数据概览 ===")
    print(f"V值数据形状: {valence_df.shape}")
    print(f"A值数据形状: {arousal_df.shape}")

    # 统计完全一致的样本
    v_consistent_count = sum(1 for _, row in valence_df.iterrows() if len(set(row)) == 1)
    a_consistent_count = sum(1 for _, row in arousal_df.iterrows() if len(set(row)) == 1)
    both_consistent_count = sum(1 for i, (_, v_row) in enumerate(valence_df.iterrows()) if len(set(v_row)) == 1 and len(set(arousal_df.iloc[i])) == 1)

    print(f"\n=== 一致性统计 ===")
    print(f"V值完全一致的样本: {v_consistent_count} 个 ({v_consistent_count / len(valence_df) * 100:.1f}%)")
    print(f"A值完全一致的样本: {a_consistent_count} 个 ({a_consistent_count / len(arousal_df) * 100:.1f}%)")
    print(f"V值和A值都完全一致的样本: {both_consistent_count} 个 ({both_consistent_count / len(valence_df) * 100:.1f}%)")

    # 描述性统计
    print(f"\n=== V值描述性统计 ===")
    print(valence_df.describe())

    print(f"\n=== A值描述性统计 ===")
    print(arousal_df.describe())

    return valence_df, arousal_df


def calculate_cronbachs_alpha_with_pingouin(df, data_type):
    """使用pingouin库计算Cronbach's Alpha"""
    try:
        # 使用pingouin计算Cronbach's Alpha
        alpha_result = pg.cronbach_alpha(df)

        print(f"\n=== {data_type} Cronbach's Alpha (Pingouin) ===")
        print(f"Cronbach's Alpha: {alpha_result[0]:.4f}")
        print(f"95% 置信区间: [{alpha_result[1][0]:.4f}, {alpha_result[1][1]:.4f}]")

        return alpha_result[0]

    except Exception as e:
        print(f"计算{data_type} Cronbach's Alpha时出错: {e}")
        return None


def calculate_cronbachs_alpha_manual(df, data_type):
    """手动计算Cronbach's Alpha作为对比"""
    data_matrix = df.values
    n_items = data_matrix.shape[1]

    # 计算项目方差之和
    item_variances = np.var(data_matrix, axis=0, ddof=1)
    total_item_variance = np.sum(item_variances)

    # 计算总分方差
    total_variance = np.var(np.sum(data_matrix, axis=1), ddof=1)

    # 计算Cronbach's Alpha
    alpha = (n_items / (n_items - 1)) * (1 - (total_item_variance / total_variance))

    print(f"\n=== {data_type} Cronbach's Alpha (手动计算) ===")
    print(f"项目数: {n_items}")
    print(f"各项目方差: {item_variances}")
    print(f"项目方差之和: {total_item_variance:.4f}")
    print(f"总分方差: {total_variance:.4f}")
    print(f"Cronbach's Alpha: {alpha:.4f}")

    return alpha


def analyze_inter_rater_reliability(valence_df, arousal_df, annotators):
    """分析标注者间信度"""
    print(f"\n=== 标注者间信度分析 ===")

    # 计算标注者间相关系数
    print(f"\nV值标注者间相关系数:")
    v_corr = valence_df.corr()
    print(v_corr.round(3))

    print(f"\nA值标注者间相关系数:")
    a_corr = arousal_df.corr()
    print(a_corr.round(3))

    # 计算组内相关系数(ICC)
    try:
        # 准备ICC分析的数据格式
        v_data_long = []
        a_data_long = []

        for idx, (audio_file, row) in enumerate(valence_df.iterrows()):
            for annotator in annotators:
                v_data_long.append({"subject": idx, "rater": annotator, "score": row[annotator]})

        for idx, (audio_file, row) in enumerate(arousal_df.iterrows()):
            for annotator in annotators:
                a_data_long.append({"subject": idx, "rater": annotator, "score": row[annotator]})

        v_icc_df = pd.DataFrame(v_data_long)
        a_icc_df = pd.DataFrame(a_data_long)

        # 计算ICC
        v_icc = pg.intraclass_corr(data=v_icc_df, targets="subject", raters="rater", ratings="score")
        a_icc = pg.intraclass_corr(data=a_icc_df, targets="subject", raters="rater", ratings="score")

        print(f"\n=== V值组内相关系数(ICC) ===")
        print(v_icc[["Type", "ICC", "F", "pval", "CI95%"]])

        print(f"\n=== A值组内相关系数(ICC) ===")
        print(a_icc[["Type", "ICC", "F", "pval", "CI95%"]])

    except Exception as e:
        print(f"计算ICC时出错: {e}")


def main():
    print("开始计算Cronbach's Alpha系数...")

    # 1. 加载标注数据
    print("加载标注数据...")
    all_data, annotators = load_annotations()

    # 2. 提取VA值
    print("提取VA值...")
    valence_df, arousal_df = extract_va_values(all_data, annotators)

    # 3. 使用pingouin计算Cronbach's Alpha
    print("\n" + "=" * 50)
    print("使用Pingouin库计算Cronbach's Alpha")
    print("=" * 50)

    valence_alpha_pg = calculate_cronbachs_alpha_with_pingouin(valence_df, "V值")
    arousal_alpha_pg = calculate_cronbachs_alpha_with_pingouin(arousal_df, "A值")

    # 4. 手动计算作为对比
    print("\n" + "=" * 50)
    print("手动计算Cronbach's Alpha (对比)")
    print("=" * 50)

    valence_alpha_manual = calculate_cronbachs_alpha_manual(valence_df, "V值")
    arousal_alpha_manual = calculate_cronbachs_alpha_manual(arousal_df, "A值")

    # 5. 标注者间信度分析
    analyze_inter_rater_reliability(valence_df, arousal_df, annotators)

    # 6. 总结
    print(f"\n" + "=" * 50)
    print("总结")
    print("=" * 50)
    if valence_alpha_pg is not None:
        print(f"V值 Cronbach's Alpha (Pingouin): {valence_alpha_pg:.4f}")
    # print(f"V值 Cronbach's Alpha (手动): {valence_alpha_manual:.4f}")

    if arousal_alpha_pg is not None:
        print(f"A值 Cronbach's Alpha (Pingouin): {arousal_alpha_pg:.4f}")
    # print(f"A值 Cronbach's Alpha (手动): {arousal_alpha_manual:.4f}")

    print("\n分析完成!")


if __name__ == "__main__":
    main()
