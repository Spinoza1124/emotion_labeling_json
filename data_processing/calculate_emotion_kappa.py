import os
import json
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from statsmodels.stats.inter_rater import fleiss_kappa
from matplotlib.gridspec import GridSpec
import pathlib

def load_labels_from_json(data_dir):
    """加载三个人的情感标签数据"""
    persons = ['huangjun', 'liuyang', 'yuhangbin']
    data = {}
    
    for person in persons:
        file_path = os.path.join(data_dir, person, 'spk77-3-1_labels.json')
        with open(file_path, 'r', encoding='utf-8') as f:
            data[person] = json.load(f)
            
    return data

def extract_emotion_labels(data):
    """提取每个音频文件的情感标签，形成对应关系表"""
    persons = list(data.keys())
    
    # 创建一个包含所有音频文件ID的集合
    all_audio_files = set()
    for person in persons:
        all_audio_files.update(item['audio_file'] for item in data[person])
    
    # 创建音频文件到各标注者标签的映射
    labels = {}
    for audio_file in all_audio_files:
        labels[audio_file] = {}
        for person in persons:
            label = next((item.get('discrete_emotion') 
                         for item in data[person] 
                         if item['audio_file'] == audio_file), None)
            # 如果标签是None(null)，使用"neutral"代替
            if label is None:
                # 检查emotion_type是否为"neutral"
                emotion_type = next((item.get('emotion_type') 
                                   for item in data[person] 
                                   if item['audio_file'] == audio_file), None)
                if emotion_type == "neutral":
                    label = "neutral"
                else:
                    label = "undefined"
            labels[audio_file][person] = label
    
    return labels

def prepare_data_for_kappa(labels):
    """准备用于计算Fleiss' Kappa的数据"""
    # 提取所有可能的情感标签类型
    all_emotions = set()
    for audio_data in labels.values():
        all_emotions.update(audio_data.values())
    
    # 将情感标签映射为数字索引（Fleiss Kappa需要）
    emotion_mapping = {emotion: idx for idx, emotion in enumerate(sorted(all_emotions))}
    
    # 创建评分矩阵：每行是一个项目，每列是一个标注类别
    # 矩阵中的值表示有多少标注者选择了该类别
    n_categories = len(emotion_mapping)
    n_items = len(labels)
    n_raters = 3  # 三个标注者
    
    # 初始化评分矩阵
    ratings_matrix = np.zeros((n_items, n_categories))
    
    # 填充评分矩阵
    for i, (_, audio_data) in enumerate(labels.items()):
        for emotion in audio_data.values():
            ratings_matrix[i, emotion_mapping[emotion]] += 1
            
    return ratings_matrix, emotion_mapping

def calculate_kappa(ratings_matrix):
    """计算Fleiss' Kappa值"""
    kappa = fleiss_kappa(ratings_matrix)
    return kappa

def calculate_agreement_per_category(labels, emotion_mapping):
    """计算每个情感类别的一致性"""
    categories = sorted(emotion_mapping.keys())
    agreement = {category: 0 for category in categories}
    counts = {category: 0 for category in categories}
    
    for audio_file, annotations in labels.items():
        # 计算此音频文件的标注
        values = list(annotations.values())
        for emotion in values:
            counts[emotion] += 1
            # 如果所有标注者都一致选择了这个情感
            if values.count(emotion) == len(annotations):
                agreement[emotion] += len(annotations)
    
    # 计算每个类别的一致性百分比
    for category in categories:
        if counts[category] > 0:
            agreement[category] = agreement[category] / counts[category] * 100
        else:
            agreement[category] = 0
            
    return agreement

def create_confusion_matrix(labels):
    """创建混淆矩阵以查看不同标注者之间的差异"""
    persons = ['huangjun', 'liuyang', 'yuhangbin']
    pairs = [('huangjun', 'liuyang'), ('huangjun', 'yuhangbin'), ('liuyang', 'yuhangbin')]
    confusion_matrices = {}
    
    # 获取所有唯一的情感标签
    all_emotions = set()
    for audio_data in labels.values():
        all_emotions.update(audio_data.values())
    all_emotions = sorted(all_emotions)
    
    # 为每对标注者创建混淆矩阵
    for p1, p2 in pairs:
        matrix = pd.DataFrame(0, index=all_emotions, columns=all_emotions)
        
        for audio_file, annotations in labels.items():
            if p1 in annotations and p2 in annotations:
                emotion1 = annotations[p1]
                emotion2 = annotations[p2]
                matrix.loc[emotion1, emotion2] += 1
                
        confusion_matrices[(p1, p2)] = matrix
        
    return confusion_matrices, all_emotions

def visualize_results(kappa, agreement, confusion_matrices, emotions, output_dir):
    """可视化结果并保存为图像"""
    # 创建一个大的图形
    plt.figure(figsize=(20, 15))
    gs = GridSpec(2, 3, figure=plt.gcf())
    
    # 1. 显示总体Kappa值
    ax1 = plt.subplot(gs[0, 0])
    kappa_text = f"Fleiss' Kappa: {kappa:.4f}"
    kappa_interpretation = interpret_kappa(kappa)
    ax1.text(0.5, 0.5, f"{kappa_text}\n\n{kappa_interpretation}", 
             ha='center', va='center', fontsize=14)
    ax1.axis('off')
    
    # 2. 每个类别的一致性条形图
    ax2 = plt.subplot(gs[0, 1:])
    categories = list(agreement.keys())
    values = list(agreement.values())
    colors = plt.cm.viridis(np.linspace(0, 1, len(categories)))
    
    bars = ax2.bar(categories, values, color=colors)
    ax2.set_ylabel('Agreement (%)')
    ax2.set_title('Agreement Percentage by Emotion Category')
    ax2.set_ylim(0, 100)
    
    # 在条形上方添加数值标签
    for bar in bars:
        height = bar.get_height()
        ax2.text(bar.get_x() + bar.get_width()/2., height + 1,
                 f'{height:.1f}%', ha='center', va='bottom')
    
    # 3. 三对标注者的混淆矩阵
    pairs = [('huangjun', 'liuyang'), ('huangjun', 'yuhangbin'), ('liuyang', 'yuhangbin')]
    positions = [(1, 0), (1, 1), (1, 2)]
    
    for (p1, p2), pos in zip(pairs, positions):
        ax = plt.subplot(gs[pos])
        matrix = confusion_matrices[(p1, p2)]
        
        # 绘制热力图
        sns.heatmap(matrix, annot=True, fmt='d', cmap='Blues', ax=ax, cbar=False)
        ax.set_title(f'Confusion Matrix: {p1} vs {p2}')
        ax.set_xlabel(f'{p2}\'s labels')
        ax.set_ylabel(f'{p1}\'s labels')
    
    plt.tight_layout()
    
    # 保存图像
    output_path = os.path.join(output_dir, 'emotion_agreement_analysis.png')
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close()
    
    # 生成基于混淆矩阵的更详细的对比图
    plt.figure(figsize=(18, 6 * len(emotions)))
    for i, emotion in enumerate(emotions):
        for j, (p1, p2) in enumerate(pairs):
            plt.subplot(len(emotions), 3, i * 3 + j + 1)
            matrix = confusion_matrices[(p1, p2)]
            # 提取与当前情感相关的行
            row = matrix.loc[emotion]
            # 绘制条形图
            bars = plt.bar(row.index, row.values, color=plt.cm.tab10(np.linspace(0, 1, len(row))))
            
            # 添加数值标签
            for bar in bars:
                height = bar.get_height()
                if height > 0:
                    plt.text(bar.get_x() + bar.get_width()/2., height + 0.1,
                             str(int(height)), ha='center', va='bottom')
            
            plt.title(f'{p1}\'s "{emotion}" labeled as by {p2}')
            plt.xticks(rotation=45, ha='right')
            plt.ylim(0, matrix.values.max() * 1.1)
    
    plt.tight_layout()
    output_path = os.path.join(output_dir, 'emotion_disagreement_details.png')
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close()

def interpret_kappa(kappa):
    """解释Kappa值的含义"""
    if kappa < 0:
        return "Poor agreement (less than chance)"
    elif kappa < 0.2:
        return "Slight agreement"
    elif kappa < 0.4:
        return "Fair agreement"
    elif kappa < 0.6:
        return "Moderate agreement"
    elif kappa < 0.8:
        return "Substantial agreement"
    else:
        return "Almost perfect agreement"

def generate_text_report(kappa, agreement, labels, emotion_mapping, output_dir):
    """生成文本报告"""
    with open(os.path.join(output_dir, 'kappa_report.txt'), 'w') as f:
        f.write("Fleiss' Kappa Analysis for Emotion Labeling\n")
        f.write("=" * 40 + "\n\n")
        
        # 总体Kappa值
        f.write(f"Overall Fleiss' Kappa: {kappa:.4f}\n")
        f.write(f"Interpretation: {interpret_kappa(kappa)}\n\n")
        
        # 每个类别的一致性
        f.write("Agreement Percentage by Emotion Category:\n")
        for emotion, percentage in agreement.items():
            f.write(f"  {emotion}: {percentage:.1f}%\n")
        
        # 统计基本信息
        total_items = len(labels)
        fully_agreed = sum(1 for annotations in labels.values() 
                          if len(set(annotations.values())) == 1)
        
        f.write(f"\nTotal audio files: {total_items}\n")
        f.write(f"Files with full agreement: {fully_agreed} ({fully_agreed/total_items*100:.1f}%)\n")
        f.write(f"Files with disagreement: {total_items - fully_agreed} ({(total_items - fully_agreed)/total_items*100:.1f}%)\n")
        
        # 列出每种情感的出现次数
        f.write("\nEmotion distribution across all annotators:\n")
        emotion_counts = {emotion: 0 for emotion in emotion_mapping.keys()}
        for annotations in labels.values():
            for emotion in annotations.values():
                emotion_counts[emotion] += 1
        
        for emotion, count in sorted(emotion_counts.items(), key=lambda x: x[1], reverse=True):
            f.write(f"  {emotion}: {count} times ({count/(total_items*3)*100:.1f}%)\n")

def main():
    # 定义脚本名称和输出目录
    script_name = os.path.splitext(os.path.basename(__file__))[0]
    
    # 找到项目根目录
    script_dir = os.path.dirname(os.path.abspath(__file__))
    project_root = os.path.dirname(script_dir)
    
    # 创建输出目录
    output_dir = os.path.join(project_root, 'result_images', script_name)
    pathlib.Path(output_dir).mkdir(parents=True, exist_ok=True)
    
    # 数据目录
    data_dir = os.path.join(project_root, 'json')
    
    # 1. 加载数据
    print("Loading data...")
    data = load_labels_from_json(data_dir)
    
    # 2. 提取情感标签
    print("Extracting emotion labels...")
    labels = extract_emotion_labels(data)
    
    # 3. 准备Kappa计算的数据
    print("Preparing data for Kappa calculation...")
    ratings_matrix, emotion_mapping = prepare_data_for_kappa(labels)
    
    # 4. 计算Fleiss' Kappa
    print("Calculating Fleiss' Kappa...")
    kappa = calculate_kappa(ratings_matrix)
    
    # 5. 计算每个类别的一致性
    print("Calculating agreement per category...")
    agreement = calculate_agreement_per_category(labels, emotion_mapping)
    
    # 6. 创建混淆矩阵
    print("Creating confusion matrices...")
    confusion_matrices, emotions = create_confusion_matrix(labels)
    
    # 7. 可视化结果
    print("Visualizing results...")
    visualize_results(kappa, agreement, confusion_matrices, emotions, output_dir)
    
    # 8. 生成文本报告
    print("Generating text report...")
    generate_text_report(kappa, agreement, labels, emotion_mapping, output_dir)
    
    print(f"Done! Results saved to {output_dir}")

if __name__ == "__main__":
    main()