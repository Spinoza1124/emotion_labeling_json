import os
import json
from collections import Counter
import matplotlib.pyplot as plt
import numpy as np

def load_discrete_emotions(data_dir, persons):
    """加载每个人的离散情感标签数据"""
    emotion_data = {}
    for person in persons:
        path = os.path.join(data_dir, person, 'spk77-3-1_labels.json')
        with open(path, 'r', encoding='utf-8') as f:
            data = json.load(f)
            
        # 提取离散情感标签，将null替换为"neutral"
        emotions = []
        for item in data:
            emotion = item.get('discrete_emotion')
            if emotion is None:
                if item.get('emotion_type') == 'neutral':
                    emotions.append('neutral')
                else:
                    emotions.append('undefined')
            else:
                emotions.append(emotion)
                
        emotion_data[person] = Counter(emotions)
    return emotion_data

def create_consistent_color_map(all_emotions):
    """创建一致的颜色映射，确保相同情感在不同人的图表中颜色一致"""
    # 使用预定义的颜色映射
    cmap = plt.get_cmap('tab10')
    colors = {}
    
    # 确保'neutral'和'undefined'有固定颜色
    if 'neutral' in all_emotions:
        colors['neutral'] = 'lightgray'
        all_emotions.remove('neutral')
    if 'undefined' in all_emotions:
        colors['undefined'] = 'darkgray'
        all_emotions.remove('undefined')
        
    # 为其他情感分配颜色
    sorted_emotions = sorted(all_emotions)
    for i, emotion in enumerate(sorted_emotions):
        colors[emotion] = cmap(i % 10)
        
    return colors

def plot_emotion_pie(emotion_data, out_dir):
    """创建饼图并保存"""
    # 获取所有情感标签
    all_emotions = set()
    for counts in emotion_data.values():
        all_emotions.update(counts.keys())
    
    # 创建一致的颜色映射
    color_map = create_consistent_color_map(list(all_emotions))
    
    # 为每个人创建饼图
    fig, axes = plt.subplots(1, len(emotion_data), figsize=(5 * len(emotion_data), 6))
    
    for i, (person, counts) in enumerate(emotion_data.items()):
        ax = axes[i]
        
        # 确保所有情感都在图表中表示
        labels = []
        sizes = []
        colors = []
        
        # 按照频次排序（从高到低）
        for emotion, count in sorted(counts.items(), key=lambda x: x[1], reverse=True):
            labels.append(f"{emotion} ({count})")
            sizes.append(count)
            colors.append(color_map[emotion])
        
        # 创建饼图
        ax.pie(sizes, labels=None, colors=colors, autopct='%1.1f%%',
               startangle=90, pctdistance=0.85)
        
        # 添加图例
        ax.legend(labels, loc="best", fontsize=10)
        ax.set_title(f"{person.capitalize()}'s Emotion Distribution")
    
    # 调整布局并保存
    plt.tight_layout()
    plt.savefig(os.path.join(out_dir, 'discrete_emotion_distribution.png'), dpi=150)
    plt.close()

def main():
    # 创建输出目录
    script_name = os.path.splitext(os.path.basename(__file__))[0]
    # 找到项目根目录
    script_dir = os.path.dirname(os.path.abspath(__file__))
    project_root = os.path.dirname(script_dir)
    
    out_dir = os.path.join(project_root, 'result_images', script_name)
    os.makedirs(out_dir, exist_ok=True)
    
    # 数据目录
    data_dir = os.path.join(project_root, 'json')
    persons = ['huangjun', 'liuyang', 'yuhangbin']
    
    # 加载情感数据
    emotion_data = load_discrete_emotions(data_dir, persons)
    
    # 创建并保存饼图
    plot_emotion_pie(emotion_data, out_dir)
    
    print(f"情感分布饼图已保存到 {out_dir}")

if __name__ == '__main__':
    main()