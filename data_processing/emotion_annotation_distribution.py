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


def analyze_emotion_data():
    """Analyze v_value and a_value distribution for three annotators using only common files"""

    # Define the base path and annotators
    base_path = "/mnt/shareEEx/liuyang/code/emotion_labeling_json/json"
    annotators = ["huangjun", "liuyang", "yuhangbin"]

    # Find common files
    common_files = find_common_files()
    if not common_files:
        print("No common files found!")
        return {}

    # Data structure to store statistics
    data_stats = {annotator: {"v_value": defaultdict(int), "a_value": defaultdict(int)} for annotator in annotators}

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

                # Count v_value and a_value distributions
                for item in data:
                    v_val = item.get("v_value")
                    a_val = item.get("a_value")

                    if v_val is not None:
                        data_stats[annotator]["v_value"][v_val] += 1
                    if a_val is not None:
                        data_stats[annotator]["a_value"][a_val] += 1

            except Exception as e:
                print(f"Error processing {filepath}: {e}")

    return data_stats


def plot_distribution(data_stats):
    """Create bar plots for v_value and a_value distributions with value labels"""

    # Set up the plot
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 7))

    # Colors for different annotators
    colors = ["#FF6B6B", "#4ECDC4", "#45B7D1"]
    annotators = list(data_stats.keys())

    # Plot v_value distribution
    v_values = set()
    a_values = set()

    # Collect all possible values
    for annotator in annotators:
        v_values.update(data_stats[annotator]["v_value"].keys())
        a_values.update(data_stats[annotator]["a_value"].keys())

    v_values = sorted(list(v_values))
    a_values = sorted(list(a_values))

    # V-value plot
    x_pos_v = np.arange(len(v_values))
    width = 0.25

    for i, annotator in enumerate(annotators):
        v_counts = [data_stats[annotator]["v_value"][v] for v in v_values]
        bars = ax1.bar(x_pos_v + i * width, v_counts, width, label=annotator.title(), color=colors[i], alpha=0.8)

        # Add value labels on bars
        for bar, count in zip(bars, v_counts):
            if count > 0:  # Only show label if count > 0
                height = bar.get_height()
                ax1.text(bar.get_x() + bar.get_width() / 2.0, height + max(v_counts) * 0.01, f"{count}", ha="center", va="bottom", fontsize=10, fontweight="bold")

    ax1.set_xlabel("V-Value (Valence)", fontsize=12)
    ax1.set_ylabel("Frequency", fontsize=12)
    ax1.set_title("Distribution of V-Value (Valence) by Annotator\n(Common Files Only)", fontsize=14, fontweight="bold")
    ax1.set_xticks(x_pos_v + width)
    ax1.set_xticklabels(v_values)
    ax1.legend()
    ax1.grid(axis="y", alpha=0.3)
    ax1.set_ylim(0, max([max(data_stats[ann]["v_value"].values(), default=0) for ann in annotators]) * 1.15)

    # A-value plot
    x_pos_a = np.arange(len(a_values))

    for i, annotator in enumerate(annotators):
        a_counts = [data_stats[annotator]["a_value"][a] for a in a_values]
        bars = ax2.bar(x_pos_a + i * width, a_counts, width, label=annotator.title(), color=colors[i], alpha=0.8)

        # Add value labels on bars
        for bar, count in zip(bars, a_counts):
            if count > 0:  # Only show label if count > 0
                height = bar.get_height()
                ax2.text(bar.get_x() + bar.get_width() / 2.0, height + max(a_counts) * 0.01, f"{count}", ha="center", va="bottom", fontsize=10, fontweight="bold")

    ax2.set_xlabel("A-Value (Arousal)", fontsize=12)
    ax2.set_ylabel("Frequency", fontsize=12)
    ax2.set_title("Distribution of A-Value (Arousal) by Annotator\n(Common Files Only)", fontsize=14, fontweight="bold")
    ax2.set_xticks(x_pos_a + width)
    ax2.set_xticklabels(a_values)
    ax2.legend()
    ax2.grid(axis="y", alpha=0.3)
    ax2.set_ylim(0, max([max(data_stats[ann]["a_value"].values(), default=0) for ann in annotators]) * 1.15)

    plt.tight_layout()
    plt.savefig("result_images/emotion_annotation_distribution.png", dpi=300, bbox_inches="tight")
    plt.show()

    # Print statistics
    print("\n" + "=" * 60)
    print("EMOTION ANNOTATION STATISTICS (COMMON FILES ONLY)")
    print("=" * 60)

    for annotator in annotators:
        total_annotations = sum(data_stats[annotator]["v_value"].values())
        print(f"\n{annotator.upper()}:")
        print(f"  Total annotations: {total_annotations}")

        print("  V-Value distribution:")
        for v_val in sorted(data_stats[annotator]["v_value"].keys()):
            count = data_stats[annotator]["v_value"][v_val]
            percentage = (count / total_annotations) * 100 if total_annotations > 0 else 0
            print(f"    V={v_val}: {count} ({percentage:.1f}%)")

        print("  A-Value distribution:")
        for a_val in sorted(data_stats[annotator]["a_value"].keys()):
            count = data_stats[annotator]["a_value"][a_val]
            percentage = (count / total_annotations) * 100 if total_annotations > 0 else 0
            print(f"    A={a_val}: {count} ({percentage:.1f}%)")


if __name__ == "__main__":
    # Analyze the data
    print("Analyzing emotion annotation data for common files...")
    stats = analyze_emotion_data()

    if stats:
        # Create visualizations
        print("Creating visualizations...")
        plot_distribution(stats)
        print("\nAnalysis complete! Chart saved as 'emotion_annotation_distribution_common.png'")
    else:
        print("No data to visualize.")
