#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
过滤JSON文件中的patient数据（只处理所有文件夹都有的共同文件）
根据patient_status字段，只保留值为"patient"的记录
"""

import json
import os
from pathlib import Path


def get_common_files(labels_dir):
    """
    获取所有子文件夹中共同存在的JSON文件

    Args:
        labels_dir (Path): labels目录路径

    Returns:
        set: 共同文件名的集合
    """
    subfolders = [subfolder for subfolder in labels_dir.iterdir() if subfolder.is_dir()]

    if not subfolders:
        return set()

    # 获取第一个文件夹的JSON文件作为基准
    common_files = set(f.name for f in subfolders[0].glob("*.json"))

    # 与其他文件夹的文件取交集
    for subfolder in subfolders[1:]:
        folder_files = set(f.name for f in subfolder.glob("*.json"))
        common_files = common_files.intersection(folder_files)

    return common_files


def filter_patient_data(input_file_path, output_file_path):
    """
    过滤JSON文件，只保留patient_status为"patient"的记录

    Args:
        input_file_path (str): 输入JSON文件路径
        output_file_path (str): 输出JSON文件路径

    Returns:
        int: 过滤后保留的记录数量
    """
    try:
        # 读取原始JSON文件
        with open(input_file_path, "r", encoding="utf-8") as f:
            data = json.load(f)

        # 过滤数据，只保留patient_status为"patient"的记录
        filtered_data = [record for record in data if record.get("patient_status") == "patient"]

        # 确保输出目录存在
        os.makedirs(os.path.dirname(output_file_path), exist_ok=True)

        # 写入过滤后的数据
        with open(output_file_path, "w", encoding="utf-8") as f:
            json.dump(filtered_data, f, ensure_ascii=False, indent=2)

        print(f"处理完成: {input_file_path}")
        print(f"  原始记录数: {len(data)}")
        print(f"  过滤后记录数: {len(filtered_data)}")
        print(f"  输出文件: {output_file_path}")
        print()

        return len(filtered_data)

    except Exception as e:
        print(f"处理文件 {input_file_path} 时出错: {str(e)}")
        return 0


def process_common_json_files():
    """
    只处理所有文件夹中都存在的共同JSON文件
    """
    # 定义路径
    labels_dir = Path("/mnt/shareEEx/liuyang/code/emotion_labeling_json/json/labels")
    new_labels_dir = Path("/mnt/shareEEx/liuyang/code/emotion_labeling_json/json/new_labels")

    # 确保new_labels目录存在
    new_labels_dir.mkdir(exist_ok=True)

    # 获取共同文件
    common_files = get_common_files(labels_dir)

    if not common_files:
        print("没有找到共同的JSON文件")
        return

    print(f"找到共同文件: {sorted(common_files)}")
    print("=" * 50)

    total_files = 0
    total_records_before = 0
    total_records_after = 0

    # 遍历所有子文件夹
    for subfolder in labels_dir.iterdir():
        if subfolder.is_dir():
            print(f"处理文件夹: {subfolder.name}")

            # 创建对应的输出子文件夹
            output_subfolder = new_labels_dir / subfolder.name
            output_subfolder.mkdir(exist_ok=True)

            # 只处理共同文件
            for filename in sorted(common_files):
                input_path = subfolder / filename
                output_path = output_subfolder / filename

                if input_path.exists():
                    # 读取原始文件获取记录数
                    try:
                        with open(input_path, "r", encoding="utf-8") as f:
                            original_data = json.load(f)
                        records_before = len(original_data)
                    except:
                        records_before = 0

                    # 过滤数据
                    records_after = filter_patient_data(str(input_path), str(output_path))

                    total_files += 1
                    total_records_before += records_before
                    total_records_after += records_after
                else:
                    print(f"警告: 文件 {input_path} 不存在")

    print("=" * 50)
    print("处理完成统计:")
    print(f"共同文件列表: {sorted(common_files)}")
    print(f"总共处理文件数: {total_files}")
    print(f"原始总记录数: {total_records_before}")
    print(f"过滤后总记录数: {total_records_after}")
    print(f"过滤掉的记录数: {total_records_before - total_records_after}")
    print(f"保留比例: {total_records_after / total_records_before * 100:.2f}%" if total_records_before > 0 else "保留比例: 0%")


if __name__ == "__main__":
    process_common_json_files()
