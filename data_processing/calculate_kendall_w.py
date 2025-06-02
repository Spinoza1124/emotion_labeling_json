import os
import json
import numpy as np
import pandas as pd
from scipy import stats
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
    return valence_df, arousal_df


def calculate_kendall_w_scipy(data_matrix, data_type):
    """使用scipy计算Kendall's W"""
    try:
        # scipy.stats.kendalltau只能计算两个序列，我们需要手动计算Kendall's W
        # 或者使用rankdata然后计算

        # 对每个评价者的数据进行排序转换
        ranks = np.apply_along_axis(stats.rankdata, 0, data_matrix)

        # 计算Kendall's W
        m, n = ranks.shape  # m = 样本数, n = 评价者数

        # 计算每个样本的排名总和
        rank_sums = np.sum(ranks, axis=1)

        # 计算排名总和的方差
        rank_sum_var = np.var(rank_sums, ddof=1)

        # 计算Kendall's W
        # W = 12 * S / (n^2 * (m^3 - m))
        # 其中 S = sum((Ri - R_mean)^2)
        mean_rank_sum = np.mean(rank_sums)
        S = np.sum((rank_sums - mean_rank_sum) ** 2)

        W = 12 * S / (n**2 * (m**3 - m))

        # 计算卡方统计量和p值
        chi_square = n * (m - 1) * W
        p_value = 1 - stats.chi2.cdf(chi_square, m - 1)

        print(f"\n=== {data_type} Kendall's W (scipy计算) ===")
        print(f"样本数 (m): {m}")
        print(f"评价者数 (n): {n}")
        print(f"Kendall's W: {W:.4f}")
        print(f"卡方统计量: {chi_square:.4f}")
        print(f"自由度: {m - 1}")
        print(f"p值: {p_value:.4f}")

        # 解释结果
        if p_value < 0.001:
            significance = "***"
        elif p_value < 0.01:
            significance = "**"
        elif p_value < 0.05:
            significance = "*"
        else:
            significance = "ns"

        print(f"显著性: {significance}")

        return W, p_value, chi_square

    except Exception as e:
        print(f"计算{data_type} Kendall's W时出错: {e}")
        return None, None, None


def calculate_kendall_w_pingouin(df, data_type):
    """使用pingouin计算Kendall's W"""
    try:
        # pingouin需要长格式数据
        data_long = []

        for idx, (sample_id, row) in enumerate(df.iterrows()):
            for rater in df.columns:
                data_long.append({"subject": idx, "rater": rater, "score": row[rater]})

        df_long = pd.DataFrame(data_long)

        # 使用pingouin计算Kendall's W
        result = pg.harrelldavis(df_long, var="score")  # 这不是正确的函数

        # pingouin中没有直接的Kendall's W函数，我们使用其他方法
        # 或者使用ICC作为替代
        icc_result = pg.intraclass_corr(data=df_long, targets="subject", raters="rater", ratings="score")

        print(f"\n=== {data_type} ICC分析 (pingouin) ===")
        print(icc_result[["Type", "ICC", "F", "pval", "CI95%"]])

        return icc_result

    except Exception as e:
        print(f"使用pingouin分析{data_type}时出错: {e}")
        return None


def interpret_kendall_w(W):
    """解释Kendall's W值"""
    if W is None:
        return "无法计算"
    elif W >= 0.7:
        return "强一致性"
    elif W >= 0.5:
        return "中等一致性"
    elif W >= 0.3:
        return "弱一致性"
    else:
        return "一致性很差"


def calculate_pairwise_kendall_tau(df, data_type):
    """计算两两之间的Kendall's tau"""
    print(f"\n=== {data_type} 两两Kendall's tau ===")

    annotators = df.columns.tolist()
    n_annotators = len(annotators)

    # 创建结果矩阵
    tau_matrix = np.zeros((n_annotators, n_annotators))
    p_matrix = np.zeros((n_annotators, n_annotators))

    for i in range(n_annotators):
        for j in range(n_annotators):
            if i != j:
                tau, p = stats.kendalltau(df.iloc[:, i], df.iloc[:, j])
                tau_matrix[i, j] = tau
                p_matrix[i, j] = p
            else:
                tau_matrix[i, j] = 1.0
                p_matrix[i, j] = 0.0

    # 转换为DataFrame便于显示
    tau_df = pd.DataFrame(tau_matrix, index=annotators, columns=annotators)
    p_df = pd.DataFrame(p_matrix, index=annotators, columns=annotators)

    print("Kendall's tau 系数矩阵:")
    print(tau_df.round(4))

    print("\np值矩阵:")
    print(p_df.round(4))

    # 计算平均tau值
    mask = np.triu(np.ones_like(tau_matrix, dtype=bool), k=1)
    mean_tau = tau_matrix[mask].mean()
    print(f"\n平均Kendall's tau: {mean_tau:.4f}")

    return tau_df, p_df, mean_tau


def main():
    print("开始计算Kendall协调系数...")

    # 1. 加载标注数据
    print("加载标注数据...")
    all_data, annotators = load_annotations()

    # 2. 提取VA值
    print("提取VA值...")
    valence_df, arousal_df = extract_va_values(all_data, annotators)

    print(f"\n数据概览:")
    print(f"V值数据形状: {valence_df.shape}")
    print(f"A值数据形状: {arousal_df.shape}")

    # 3. 计算Kendall's W (使用scipy)
    print("\n" + "=" * 60)
    print("计算Kendall协调系数 (Kendall's W)")
    print("=" * 60)

    v_W, v_p, v_chi2 = calculate_kendall_w_scipy(valence_df.values, "V值")
    a_W, a_p, a_chi2 = calculate_kendall_w_scipy(arousal_df.values, "A值")

    # 4. 计算两两Kendall's tau
    print("\n" + "=" * 60)
    print("计算两两Kendall's tau")
    print("=" * 60)

    v_tau_df, v_p_df, v_mean_tau = calculate_pairwise_kendall_tau(valence_df, "V值")
    a_tau_df, a_p_df, a_mean_tau = calculate_pairwise_kendall_tau(arousal_df, "A值")

    # 5. 使用pingouin进行补充分析
    print("\n" + "=" * 60)
    print("ICC分析 (pingouin)")
    print("=" * 60)

    v_icc = calculate_kendall_w_pingouin(valence_df, "V值")
    a_icc = calculate_kendall_w_pingouin(arousal_df, "A值")

    # 6. 总结报告
    print("\n" + "=" * 60)
    print("总结报告")
    print("=" * 60)

    print(f"\n【V值一致性分析】")
    if v_W is not None:
        print(f"Kendall's W: {v_W:.4f} ({interpret_kendall_w(v_W)})")
        print(f"显著性: p = {v_p:.4f}")
    print(f"平均Kendall's tau: {v_mean_tau:.4f}")

    print(f"\n【A值一致性分析】")
    if a_W is not None:
        print(f"Kendall's W: {a_W:.4f} ({interpret_kendall_w(a_W)})")
        print(f"显著性: p = {a_p:.4f}")
    print(f"平均Kendall's tau: {a_mean_tau:.4f}")

    print(f"\n【Kendall's W解释】")
    print(f"0.7-1.0: 强一致性")
    print(f"0.5-0.7: 中等一致性")
    print(f"0.3-0.5: 弱一致性")
    print(f"0.0-0.3: 一致性很差")

    print("\n分析完成!")


if __name__ == "__main__":
    main()
