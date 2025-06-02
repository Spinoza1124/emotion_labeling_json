import json
import os
import matplotlib.pyplot as plt
import numpy as np
from collections import defaultdict


def find_common_files():
    """Find files that exist for all three annotators"""
    base_path = "/mnt/shareEEx/liuyang/code/emotion_labeling_json/json"
    annotators = ["huangjun", "liuyang", "yuhangbin"]

    # Get files for each annotator
    annotator_files = {}
    for annotator in annotators:
        annotator_path = os.path.join(base_path, annotator)
        if os.path.exists(annotator_path):
            files = [f for f in os.listdir(annotator_path) if f.endswith("_labels.json")]
            annotator_files[annotator] = set(files)
        else:
            print(f"Warning: Path {annotator_path} does not exist")
            annotator_files[annotator] = set()

    # Find intersection of all files
    if annotator_files:
        common_files = set.intersection(*annotator_files.values())
        print(f"Found {len(common_files)} common files across all annotators:")
        for file in sorted(common_files):
            print(f"  - {file}")
        return common_files
    else:
        return set()


def analyze_discrete_emotion_data():
    """Analyze discrete emotion distribution for three annotators using only common files"""

    # Define the base path and annotators
    base_path = "/mnt/shareEEx/liuyang/code/emotion_labeling_json/json"
    annotators = ["huangjun", "liuyang", "yuhangbin"]

    # Find common files
    common_files = find_common_files()
    if not common_files:
        print("No common files found!")
        return {}

    # Data structure to store statistics
    data_stats = {annotator: {"discrete_emotion": defaultdict(int)} for annotator in annotators}

    # Process each annotator's common files
    for annotator in annotators:
        annotator_path = os.path.join(base_path, annotator)
        if not os.path.exists(annotator_path):
            print(f"Warning: Path {annotator_path} does not exist")
            continue

        # Process only common files for this annotator
        for filename in common_files:
            filepath = os.path.join(annotator_path, filename)
            try:
                with open(filepath, "r", encoding="utf-8") as f:
                    data = json.load(f)

                # Count discrete emotion distributions
                for item in data:
                    discrete_emotion = item.get("discrete_emotion")

                    # Handle null values and convert to string for consistency
                    if discrete_emotion is None:
                        discrete_emotion = "None"
                    elif isinstance(discrete_emotion, str):
                        discrete_emotion = discrete_emotion.strip()
                        if discrete_emotion == "" or discrete_emotion.lower() == "null":
                            discrete_emotion = "None"

                    data_stats[annotator]["discrete_emotion"][discrete_emotion] += 1

            except Exception as e:
                print(f"Error processing {filepath}: {e}")

    return data_stats


def plot_discrete_emotion_distribution(data_stats):
    """Create bar plot for discrete emotion distribution with value labels"""

    # Set up the plot
    fig, ax = plt.subplots(1, 1, figsize=(14, 8))

    # Colors for different annotators
    colors = ["#FF6B6B", "#4ECDC4", "#45B7D1"]
    annotators = list(data_stats.keys())

    # Collect all possible discrete emotions
    all_emotions = set()
    for annotator in annotators:
        all_emotions.update(data_stats[annotator]["discrete_emotion"].keys())

    # Sort emotions (put "None" at the end for better visualization)
    emotions = sorted([e for e in all_emotions if e != "None"])
    if "None" in all_emotions:
        emotions.append("None")

    # Discrete emotion plot
    x_pos = np.arange(len(emotions))
    width = 0.25

    for i, annotator in enumerate(annotators):
        emotion_counts = [data_stats[annotator]["discrete_emotion"][emotion] for emotion in emotions]
        bars = ax.bar(x_pos + i * width, emotion_counts, width, label=annotator.title(), color=colors[i], alpha=0.8)

        # Add value labels on bars
        for bar, count in zip(bars, emotion_counts):
            if count > 0:  # Only show label if count > 0
                height = bar.get_height()
                ax.text(bar.get_x() + bar.get_width() / 2.0, height + max(emotion_counts) * 0.01, f"{count}", ha="center", va="bottom", fontsize=10, fontweight="bold")

    ax.set_xlabel("Discrete Emotion", fontsize=12)
    ax.set_ylabel("Frequency", fontsize=12)
    ax.set_title("Distribution of Discrete Emotions by Annotator\n(Common Files Only)", fontsize=14, fontweight="bold")
    ax.set_xticks(x_pos + width)
    ax.set_xticklabels(emotions, rotation=45, ha="right")
    ax.legend()
    ax.grid(axis="y", alpha=0.3)

    # Set y-axis limit to accommodate labels
    max_count = max([max(data_stats[ann]["discrete_emotion"].values(), default=0) for ann in annotators])
    ax.set_ylim(0, max_count * 1.15)

    plt.tight_layout()
    plt.savefig("discrete_emotion_distribution.png", dpi=300, bbox_inches="tight")
    plt.show()

    # Print statistics
    print("\n" + "=" * 70)
    print("DISCRETE EMOTION ANNOTATION STATISTICS (COMMON FILES ONLY)")
    print("=" * 70)

    for annotator in annotators:
        total_annotations = sum(data_stats[annotator]["discrete_emotion"].values())
        print(f"\n{annotator.upper()}:")
        print(f"  Total annotations: {total_annotations}")

        print("  Discrete emotion distribution:")
        for emotion in sorted(data_stats[annotator]["discrete_emotion"].keys()):
            count = data_stats[annotator]["discrete_emotion"][emotion]
            percentage = (count / total_annotations) * 100 if total_annotations > 0 else 0
            print(f"    {emotion}: {count} ({percentage:.1f}%)")


def create_emotion_summary_table(data_stats):
    """Create a summary table showing emotion distribution across annotators"""

    # Collect all emotions
    all_emotions = set()
    annotators = list(data_stats.keys())

    for annotator in annotators:
        all_emotions.update(data_stats[annotator]["discrete_emotion"].keys())

    emotions = sorted([e for e in all_emotions if e != "None"])
    if "None" in all_emotions:
        emotions.append("None")

    print("\n" + "=" * 80)
    print("EMOTION DISTRIBUTION SUMMARY TABLE")
    print("=" * 80)

    # Print header
    header = f"{'Emotion':<20}"
    for annotator in annotators:
        header += f"{annotator.title():<15}"
    print(header)
    print("-" * 80)

    # Print data rows
    for emotion in emotions:
        row = f"{emotion:<20}"
        for annotator in annotators:
            count = data_stats[annotator]["discrete_emotion"][emotion]
            total = sum(data_stats[annotator]["discrete_emotion"].values())
            percentage = (count / total) * 100 if total > 0 else 0
            row += f"{count} ({percentage:.1f}%)<15"
        print(row)


if __name__ == "__main__":
    # Analyze the data
    print("Analyzing discrete emotion annotation data for common files...")
    stats = analyze_discrete_emotion_data()

    if stats:
        # Create visualizations
        print("Creating visualizations...")
        plot_discrete_emotion_distribution(stats)

        # Create summary table
        create_emotion_summary_table(stats)

        print("\nAnalysis complete! Chart saved as 'discrete_emotion_distribution.png'")
    else:
        print("No data to visualize.")
