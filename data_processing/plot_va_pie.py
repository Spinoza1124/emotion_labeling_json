import os
import json
from collections import Counter
import matplotlib.pyplot as plt

def load_counts(data_dir, persons, key):
    """按 person 统计 key 的取值频次"""
    counts = {}
    for p in persons:
        path = os.path.join(data_dir, p, 'spk77-3-1_labels.json')
        with open(path, 'r', encoding='utf-8') as f:
            data = json.load(f)
        cnt = Counter(item.get(key) for item in data)
        counts[p] = cnt
    return counts

def make_pie_figure(counts, title, out_path):
    # 收集所有可能的标签并生成一致的配色
    labels = sorted({v for cnt in counts.values() for v in cnt})
    cmap = plt.get_cmap('tab10')
    color_map = {lab: cmap(i) for i, lab in enumerate(labels)}
    # 绘制 1×3 子图
    fig, axes = plt.subplots(1, len(counts), figsize=(5*len(counts), 5))
    for ax, (p, cnt) in zip(axes, counts.items()):
        sizes = [cnt.get(lab, 0) for lab in labels]
        ax.pie(sizes, labels=labels, colors=[color_map[l] for l in labels],
               autopct='%1.1f%%', textprops={'fontsize': 10})
        ax.set_title(p)
    fig.suptitle(title, fontsize=16)
    fig.tight_layout(rect=[0, 0, 1, 0.95])
    fig.savefig(out_path, dpi=150)
    plt.close(fig)

def main():
    script_name = os.path.splitext(os.path.basename(__file__))[0]
    # 准备输出目录
    out_root = os.path.join('/mnt/shareEEx/liuyang/code/emotion_labeling_json/result_images', script_name)
    os.makedirs(out_root, exist_ok=True)

    # --- 修改开始：根据脚本路径动态定位项目根目录下的 json 文件夹 ---
    script_dir = os.path.dirname(os.path.abspath(__file__))
    project_root = os.path.dirname(script_dir)
    data_dir = os.path.join(project_root, 'json')
    # --- 修改结束 ---

    persons = ['huangjun', 'liuyang', 'yuhangbin']

    # 1. 绘制 v_value 饼图
    v_counts = load_counts(data_dir, persons, 'v_value')
    make_pie_figure(v_counts, 'V Value Distribution', os.path.join(out_root, 'v_value_pie.png'))

    # 2. 绘制 a_value 饼图
    a_counts = load_counts(data_dir, persons, 'a_value')
    make_pie_figure(a_counts, 'A Value Distribution', os.path.join(out_root, 'a_value_pie.png'))

if __name__ == '__main__':
    main()