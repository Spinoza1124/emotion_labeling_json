import os
import json
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy import stats
import seaborn as sns
from sklearn.metrics import r2_score
from itertools import combinations

# 获取脚本所在目录的绝对路径
script_dir = os.path.dirname(os.path.abspath(__file__))
# 获取项目根目录路径
project_root = os.path.dirname(script_dir)

def calculate_cronbachs_alpha(data_matrix):
    """
    计算Cronbach's Alpha系数
    
    参数:
    data_matrix - 形状为[n_subjects, n_items]的numpy数组
    
    返回:
    alpha - Cronbach's Alpha系数
    """
    n_items = data_matrix.shape[1]
    
    # 计算项目方差之和
    item_variances = np.var(data_matrix, axis=0, ddof=1)
    total_item_variance = np.sum(item_variances)
    
    # 计算总分方差
    total_variance = np.var(np.sum(data_matrix, axis=1), ddof=1)
    
    # 计算Cronbach's Alpha
    alpha = (n_items / (n_items - 1)) * (1 - (total_item_variance / total_variance))
    
    return alpha

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
        files = [f for f in os.listdir(annotator_dir) if f.endswith('_labels.json')]
        
        if not common_files:
            common_files = set(files)
        else:
            common_files &= set(files)
    
    # 加载共有文件的标注数据
    for filename in common_files:
        file_data = {annotator: [] for annotator in annotators}
        
        for annotator in annotators:
            file_path = os.path.join(base_dir, annotator, filename)
            with open(file_path, 'r') as f:
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
                valence_data[audio_file] = []
                arousal_data[audio_file] = []
            
            for annotator in annotators:
                # JSON中VA值存储在"v_value"和"a_value"字段中
                valence_data[audio_file].append(processed_data[annotator][audio_file].get("v_value", 0))
                arousal_data[audio_file].append(processed_data[annotator][audio_file].get("a_value", 0))
    
    # 转换为numpy数组，形状为[n_samples, n_annotators]
    valence_matrix = np.array([valence_data[audio_file] for audio_file in valence_data])
    arousal_matrix = np.array([arousal_data[audio_file] for audio_file in arousal_data])
    
    print(f"共处理了 {len(valence_data)} 个音频样本")
    
    return valence_matrix, arousal_matrix

def calculate_correlations(data_matrix, annotators):
    """计算标注者之间的相关系数"""
    n_annotators = len(annotators)
    corr_matrix = np.zeros((n_annotators, n_annotators))
    
    for i in range(n_annotators):
        for j in range(n_annotators):
            if i != j:
                corr, _ = stats.pearsonr(data_matrix[:, i], data_matrix[:, j])
                corr_matrix[i, j] = corr
            else:
                corr_matrix[i, j] = 1.0
    
    return corr_matrix

def plot_results(valence_alpha, arousal_alpha, v_corr_matrix, a_corr_matrix, annotators):
    """绘制结果柱状图和相关性热图"""
    # 创建保存结果的目录
    result_dir = os.path.join(project_root, "result_images/caculate_va_value/")
    os.makedirs(result_dir, exist_ok=True)
    
    # 1. 绘制Cronbach's Alpha的柱状图 - 分开显示V和A
    # Valence Alpha
    plt.figure(figsize=(8, 6))
    plt.bar(["Valence"], [valence_alpha], color='skyblue')
    plt.title("Cronbach's Alpha Coefficient for Valence Annotations")
    plt.ylim(0, 1)
    plt.axhline(y=0.7, linestyle='--', color='red', alpha=0.7)  # 0.7是一个常见的可接受阈值
    plt.text(0, valence_alpha + 0.02, f'{valence_alpha:.3f}', ha='center')
    plt.savefig(os.path.join(result_dir, "valence_cronbachs_alpha.png"), dpi=300, bbox_inches='tight')
    
    # Arousal Alpha
    plt.figure(figsize=(8, 6))
    plt.bar(["Arousal"], [arousal_alpha], color='lightgreen')
    plt.title("Cronbach's Alpha Coefficient for Arousal Annotations")
    plt.ylim(0, 1)
    plt.axhline(y=0.7, linestyle='--', color='red', alpha=0.7)
    plt.text(0, arousal_alpha + 0.02, f'{arousal_alpha:.3f}', ha='center')
    plt.savefig(os.path.join(result_dir, "arousal_cronbachs_alpha.png"), dpi=300, bbox_inches='tight')
    
    # 2 & 3. 提取标注者对之间的相关性 - 分开显示V和A
    pairs = []
    v_corrs = []
    a_corrs = []
    
    # 提取所有标注者对的相关系数
    for i in range(len(annotators)):
        for j in range(i+1, len(annotators)):  # 只取上三角矩阵，避免重复
            pair_name = f"{annotators[i]}-{annotators[j]}"
            pairs.append(pair_name)
            v_corrs.append(v_corr_matrix[i, j])
            a_corrs.append(a_corr_matrix[i, j])
    
    # 在柱状图上添加数值标签
    def autolabel(rects):
        for rect in rects:
            height = rect.get_height()
            ax.annotate(f'{height:.3f}',
                        xy=(rect.get_x() + rect.get_width() / 2, height),
                        xytext=(0, 3),
                        textcoords="offset points",
                        ha='center', va='bottom')
    
    # Valence相关性柱状图
    fig, ax = plt.subplots(figsize=(12, 7))
    rects = ax.bar(pairs, v_corrs, color='skyblue')
    ax.set_ylabel('Correlation Coefficient')
    ax.set_title('Valence Correlation between Annotators')
    ax.set_xticks(range(len(pairs)))
    ax.set_xticklabels(pairs, rotation=45, ha='right')
    
    autolabel(rects)
    
    plt.tight_layout()
    plt.savefig(os.path.join(result_dir, "valence_pairwise_correlation.png"), dpi=300, bbox_inches='tight')
    
    # Arousal相关性柱状图
    fig, ax = plt.subplots(figsize=(12, 7))
    rects = ax.bar(pairs, a_corrs, color='lightgreen')
    ax.set_ylabel('Correlation Coefficient')
    ax.set_title('Arousal Correlation between Annotators')
    ax.set_xticks(range(len(pairs)))
    ax.set_xticklabels(pairs, rotation=45, ha='right')
    
    autolabel(rects)
    
    plt.tight_layout()
    plt.savefig(os.path.join(result_dir, "arousal_pairwise_correlation.png"), dpi=300, bbox_inches='tight')
    
    # 4. 绘制平均相关系数的柱状图 - 分开显示V和A
    # 计算每个标注者与其他标注者的平均相关系数
    v_avg_corr = []
    a_avg_corr = []
    
    for i in range(len(annotators)):
        # 排除自身的相关系数(1.0)
        v_others_corr = [v_corr_matrix[i][j] for j in range(len(annotators)) if i != j]
        a_others_corr = [a_corr_matrix[i][j] for j in range(len(annotators)) if i != j]
        
        v_avg_corr.append(np.mean(v_others_corr))
        a_avg_corr.append(np.mean(a_others_corr))
    
    # Valence平均相关性
    fig, ax = plt.subplots(figsize=(10, 7))
    rects = ax.bar(annotators, v_avg_corr, color='skyblue')
    
    ax.set_ylabel('Average Correlation')
    ax.set_title('Average Valence Correlation by Annotator')
    ax.set_xticks(range(len(annotators)))
    ax.set_xticklabels(annotators)
    
    autolabel(rects)
    
    plt.tight_layout()
    plt.savefig(os.path.join(result_dir, "valence_avg_correlation_by_annotator.png"), dpi=300, bbox_inches='tight')
    
    # Arousal平均相关性
    fig, ax = plt.subplots(figsize=(10, 7))
    rects = ax.bar(annotators, a_avg_corr, color='lightgreen')
    
    ax.set_ylabel('Average Correlation')
    ax.set_title('Average Arousal Correlation by Annotator')
    ax.set_xticks(range(len(annotators)))
    ax.set_xticklabels(annotators)
    
    autolabel(rects)
    
    plt.tight_layout()
    plt.savefig(os.path.join(result_dir, "arousal_avg_correlation_by_annotator.png"), dpi=300, bbox_inches='tight')
    
    print(f"分析完成! 结果已保存到 {result_dir} 文件夹")

def main():
    # 1. 加载标注数据
    print("加载标注数据...")
    all_data, annotators = load_annotations()
    
    # 2. 提取VA值
    print("提取VA值...")
    valence_matrix, arousal_matrix = extract_va_values(all_data, annotators)
    
    # 3. 计算Cronbach's Alpha
    print("计算Cronbach's Alpha...")
    valence_alpha = calculate_cronbachs_alpha(valence_matrix)
    arousal_alpha = calculate_cronbachs_alpha(arousal_matrix)
    
    print(f"Valence Cronbach's Alpha: {valence_alpha:.3f}")
    print(f"Arousal Cronbach's Alpha: {arousal_alpha:.3f}")
    
    # 4. 计算相关性
    print("计算标注者之间的相关性...")
    v_corr_matrix = calculate_correlations(valence_matrix, annotators)
    a_corr_matrix = calculate_correlations(arousal_matrix, annotators)
    
    # 5. 绘制结果
    print("绘制结果...")
    plot_results(valence_alpha, arousal_alpha, v_corr_matrix, a_corr_matrix, annotators)

if __name__ == "__main__":
    main()