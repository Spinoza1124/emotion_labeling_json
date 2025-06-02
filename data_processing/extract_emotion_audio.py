import os
import json
import shutil
from typing import List


def find_person_folders(json_path: str) -> List[str]:
    """
    找出json文件夹下所有人员

    Args:
        json_path (str): json文件路径

    Returns:
        List[str]: 返回人员的列表
    """
    person_folders = []
    for item in os.listdir(json_path):
        item_path = os.path.join(json_path, item)
        if os.path.isdir(item_path):
            person_folders.append(item_path)
    return person_folders


def process_json_discrete_emotion(person_json_path: str, wav_path: str, output_base_dir: str) -> None:
    """
    处理单个Json文件，按emotion_type提取音频文件并复制到对应分数目录

    Args:
        person_json_path (str): 每个人的json文件的路径
        wav_path (str): wav文件的路径
        output_base_dir (str): 处理结果保存路径
    """
    try:
        with open(person_json_path, "r", encoding="utf-8") as f:
            annotations = json.load(f)
    except json.JSONDecodeError as e:
        print(f"警告：json文件 {json_path} 解析失败，错误：{str(e)}")
        return

    for item in annotations:
        emotion_type = item.get("discrete_emotion")
        audio_file = item.get("audio_file")

        # 跳过无效数据
        if not audio_file or emotion_type is None:
            continue
        if not isinstance(emotion_type, (str)):
            continue

        # 构建源文件路径
        source_path = os.path.join(wav_path, os.path.basename(person_json_path).split("_")[0], audio_file)
        if not os.path.exists(source_path):
            print(f"警告：音频文件 {source_path} 不存在")
            continue

        if emotion_type == "null":
            discrete_emotion = "neutral"
        else:
            discrete_emotion = emotion_type

        # 构建目标目录（按v_value创建子目录)
        target_dir = os.path.join(output_base_dir, os.path.basename(person_json_path).split("_")[0], "discrete_emotion", f"{discrete_emotion}")
        os.makedirs(target_dir, exist_ok=True)

        # 复制文件(使用copy2保留元数据)
        target_path = os.path.join(target_dir, audio_file)
        if os.path.exists(target_path):
            return False
        else:
            try:
                shutil.copy2(source_path, target_path)
                print(f"已复制：{audio_file} -> {target_path}")
            except Exception as e:
                print(f"复制文件错误：{audio_file}, 错误：{e}")

def process_json_v_value(person_json_path: str, wav_path: str, output_base_dir: str) -> None:
    """
    处理单个Json文件，按v_value的分数提取音频文件并复制到对应分数目录

    Args:
        person_json_path (str): 每个人的json文件的路径
        wav_path (str): wav文件的路径
        output_base_dir (str): 处理结果保存路径
    """
    try:
        with open(person_json_path, "r", encoding="utf-8") as f:
            annotations = json.load(f)
    except json.JSONDecodeError as e:
        print(f"警告：json文件 {json_path} 解析失败，错误：{str(e)}")
        return

    for item in annotations:
        v_value = item.get("v_value")
        audio_file = item.get("audio_file")

        # 跳过无效数据
        if not audio_file or v_value is None:
            continue
        if not isinstance(v_value, (int, float)):
            continue

        # 构建源文件路径
        source_path = os.path.join(wav_path, os.path.basename(person_json_path).split("_")[0], audio_file)
        if not os.path.exists(source_path):
            print(f"警告：音频文件 {source_path} 不存在")
            continue

        # 构建目标目录（按v_value创建子目录)
        target_dir = os.path.join(output_base_dir, os.path.basename(person_json_path).split("_")[0], "continue_emotion", f"v:{v_value}")
        os.makedirs(target_dir, exist_ok=True)

        # 复制文件(使用copy2保留元数据)
        target_path = os.path.join(target_dir, audio_file)
        if os.path.exists(target_path):
            return False
        else:
            try:
                shutil.copy2(source_path, target_path)
                print(f"已复制：{audio_file} -> {target_path}")
            except Exception as e:
                print(f"复制文件错误：{audio_file}, 错误：{e}")


def process_json_a_value(person_json_path: str, wav_path: str, output_base_dir: str) -> None:
    """
    处理单个Json文件，按a_value的分数提取音频文件并复制到对应分数目录

    Args:
        person_json_path (str): 每个人的json文件的路径
        wav_path (str): wav文件的路径
        output_base_dir (str): 处理结果保存路径
    """
    try:
        with open(person_json_path, "r", encoding="utf-8") as f:
            annotations = json.load(f)
    except json.JSONDecodeError as e:
        print(f"警告：json文件 {json_path} 解析失败，错误：{str(e)}")
        return

    for item in annotations:
        a_value = item.get("a_value")
        audio_file = item.get("audio_file")

        # 跳过无效数据
        if not audio_file or a_value is None:
            continue
        if not isinstance(a_value, (int, float)):
            continue

        # 构建源文件路径
        source_path = os.path.join(wav_path, os.path.basename(person_json_path).split("_")[0], audio_file)
        if not os.path.exists(source_path):
            print(f"警告：音频文件 {source_path} 不存在")
            continue

        # 构建目标目录（按v_value创建子目录)
        target_dir = os.path.join(output_base_dir, os.path.basename(person_json_path).split("_")[0], "continue_emotion", f"a:{a_value}")
        os.makedirs(target_dir, exist_ok=True)

        # 复制文件(使用copy2保留元数据)
        target_path = os.path.join(target_dir, audio_file)
        if os.path.exists(target_path):
            return False
        else:
            try:
                shutil.copy2(source_path, target_path)
                print(f"已复制：{audio_file} -> {target_path}")
            except Exception as e:
                print(f"复制文件错误：{audio_file}, 错误：{e}")


def process_person_folder(person_folder: str, wav_path: str, output_path: str) -> None:
    """
    处理每个人员文件夹下所有的json文件
    Args:
        person_folder (str): 每个人的json文件夹路径
        wav_path (str): wav文件的路径
        output_path (str): 处理结果保存路径
    """
    person_name = os.path.basename(person_folder)
    for root_dir, _, files in os.walk(person_folder):
        for file in files:
            if file.endswith("_labels.json"):
                person_json_path = os.path.join(root_dir, file)
                process_json_v_value(person_json_path, wav_path, os.path.join(output_path, person_name))
                process_json_a_value(person_json_path, wav_path, os.path.join(output_path, person_name))
                process_json_discrete_emotion(person_json_path, wav_path, os.path.join(output_path, person_name))


def main(json_path: str, wav_path: str, output_path: str) -> None:
    """
    处理程序主函数

    Args:
        json_path (str): json文件路径
        wav_path (str): wav文件路径
        output_path (str): 处理结果保存路径
    """

    # 找出json文件下所有人
    print("开始查找人员文件夹...")
    person_folders = find_person_folders(json_path)
    print(f"找到 {len(person_folders)} 个人员文件夹，开始处理")

    for person_folder in person_folders:
        print(f"处理人员文件夹：{person_folder}")
        process_person_folder(person_folder, wav_path, output_path)

    print(f"处理完成！符合条件的音频已经被复制到：{os.path.abspath(output_path)}")


if __name__ == "__main__":
    json_path = "/mnt/shareEEx/liuyang/code/emotion_labeling_json/json"
    wav_path = "/mnt/shareEEx/liuyang/code/emotion_labeling/emotion_annotation/"
    output_path = "/mnt/shareEEx/liuyang/code/emotion_labeling_json/extract_wav"
    main(json_path, wav_path, output_path)
