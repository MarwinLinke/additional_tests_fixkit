import pandas as pd
import matplotlib.pyplot as plt
from typing import List
from pathlib import Path
import seaborn as sns
import numpy as np

class EvalData:
    def __init__(
        self,
        label,
        frames,
        save_location
    ):
        self.label = label
        self.frames = frames
        self.save_location = save_location

def create_success_graph(subject: str, bug_id: int, type: bool, eval_datas: List[EvalData]):
    x_labels = [
        "B1-1", "B1-10", "F5-5", "F10-10", "F30-30", "F50-50",
        "V5-5", "V10-10", "V30-30", "V50-50", "C5-5", "C10-10", "C30-30", "C50-50"
    ]

    num_groups = len(eval_datas)
    group_width = 0.75
    bar_width = group_width / num_groups

    avg_stopped_repairs_matrix = []
    avg_success_rates_matrix = []
    avg_overfitting_matrix = []

    for eval_data in eval_datas:
        frames = eval_data.frames
        stopped_repairs = []
        success_rates = []
        overfitting = []
        for frame in frames:
            filtered_rows = frame[(frame['subject'] == subject) & (frame['bug_id'] == bug_id)]
            stopped_repairs.append([
                gen < 10 or (gen == 10 and score == 1.0)
                for gen, score in zip(filtered_rows["generations"].values, filtered_rows["f1_score"].values)
            ])
            success_rates.append([
                score == 1.0 for score in filtered_rows["f1_score"].values
            ])
            overfitting.append(   
                gen < 10 and score < 1.0
                for gen, score in zip(filtered_rows["generations"].values, filtered_rows["f1_score"].values)
            )

        stopped_repairs_df = pd.DataFrame(stopped_repairs).T
        avg_stopped_repairs = stopped_repairs_df.sum(axis=1) / len(frames)
        avg_stopped_repairs_matrix.append(avg_stopped_repairs)

        success_rates_df = pd.DataFrame(success_rates).T
        avg_success_rates = success_rates_df.sum(axis=1) / len(frames)
        avg_success_rates_matrix.append(avg_success_rates)

        overfitting_df = pd.DataFrame(overfitting).T
        avg_overfitting = overfitting_df.sum(axis=1) / len(frames)
        avg_overfitting_matrix.append(avg_overfitting)

    base_colors = sns.color_palette("YlGnBu", num_groups)
    #base_colors = sns.color_palette("Set2", 10)
    #base_colors = [base_colors[5]] + base_colors[:5]

    plt.figure(figsize=(12, 5))
    x = range(len(x_labels))

    if type == A:
        plot_type = avg_success_rates_matrix
        y_label = "Actual Success Rate"
        file_name = "actual"
    if type == P:
        plot_type = avg_stopped_repairs_matrix
        y_label = "Perceived Success Rate"
        file_name = "perceived"
    if type == O:
        plot_type = avg_overfitting_matrix
        y_label = "Overfitting Rate"
        file_name = "overfitting"

    for i, plot_data in enumerate(plot_type):
        offset = (i - (num_groups - 1) / 2) * bar_width
        plt.bar(
            [pos + offset for pos in x],
            plot_data,
            bar_width,
            label=f'{eval_datas[i].label}',
            color=base_colors[i],
            edgecolor='black',
            alpha=0.9,
            zorder=2,
        )
    
    plt.ylim(0, 1.5)
    plt.yticks(np.arange(0, 1.1, 0.2))  # Ticks only from 0 to 1
    plt.gca().set_yticklabels([f"{tick :.1f}" if tick <= 1 else "" for tick in plt.gca().get_yticks()])  # Hide labels > 1
    plt.grid(axis='y', linestyle='-', alpha=0.6, zorder=0)

    #plt.axvline(x = 1.5, color='grey', linewidth=0.8, linestyle='-', alpha=0.4, zorder = 0)
    #plt.axvline(x = 5.5, color='grey', linewidth=0.8, linestyle='-', alpha=0.4, zorder = 0)
    #plt.axvline(x = 9.5, color='grey', linewidth=0.8, linestyle='-', alpha=0.4, zorder = 0)

    for x_pos in [1.5, 5.5, 9.5]:  
        plt.vlines(x=x_pos, ymin=0, ymax=1, color='grey', linewidth=0.8, linestyle='-', alpha=0.4, zorder=0)

    #plt.axhline(y=1, color='black', linewidth=1, linestyle='-', zorder=3)

    plt.xlabel('Test cases (failing-passing)', fontsize=12)
    plt.ylabel(y_label, fontsize=12)
    plt.xticks(ticks=x, labels=x_labels, rotation=45, ha='right', fontsize=10)
    plt.yticks(fontsize=10)
    #plt.legend(title="Modifiers", fontsize=10, title_fontsize=12)
    plt.legend(
        title="Modifiers",
        fontsize=10,
        title_fontsize=12,
        loc="upper left",
        bbox_to_anchor=(0, 1),  # Align top-left of legend to top-left of the plot
        ncol=1  # Keep items in a single column
        )
    plt.grid(axis='y', linestyle='-', alpha=0.6, zorder = 0)
    plt.tight_layout()
    path = Path(f"plots/success_rates")
    path.mkdir(exist_ok=True, parents=True)
    plt.savefig(path / f"{file_name}_{subject}{bug_id}_mod{len(eval_datas)}.png", dpi=300)
    #plt.show()


def create_average_graph(subject: str, bug_id: int, type: bool, eval_datas: List[EvalData]):
    
    num_groups = len(eval_datas)
    bar_width = 0.4

    avg_stopped_repairs_series = []
    avg_success_rates_series = []
    avg_overfitting_series = []

    for eval_data in eval_datas:
        frames = eval_data.frames
        stopped_repairs = []
        success_rates = []
        overfitting = []
        for frame in frames:
            filtered_rows = frame[(frame['subject'] == subject) & (frame['bug_id'] == bug_id)]
            stopped_repairs.append([
                gen < 10 or (gen == 10 and score == 1.0)
                for gen, score in zip(filtered_rows["generations"].values, filtered_rows["f1_score"].values)
            ])
            success_rates.append([
                score == 1.0 for score in filtered_rows["f1_score"].values
            ])
            overfitting.append(   
                gen < 10 and score < 1.0
                for gen, score in zip(filtered_rows["generations"].values, filtered_rows["f1_score"].values)
            )

        stopped_repairs_df = pd.DataFrame(stopped_repairs).T
        avg_stopped_repairs = stopped_repairs_df.sum(axis=1) / len(frames)
        avg_stopped_repairs_series.append(avg_stopped_repairs.mean())

        success_rates_df = pd.DataFrame(success_rates).T
        avg_success_rates = success_rates_df.sum(axis=1) / len(frames)
        avg_success_rates_series.append(avg_success_rates.mean())


        overfitting_df = pd.DataFrame(overfitting).T
        avg_overfitting = overfitting_df.sum(axis=1) / len(frames)
        avg_overfitting_series.append(avg_overfitting.mean())


    #base_colors = sns.color_palette("YlGnBu", num_groups)
    base_colors = sns.color_palette("Set2", 10)
    base_colors = [base_colors[5]] + base_colors[:5]

    plt.figure(figsize=(6, 5))
    x = range(len(eval_datas))

    if type == A:
        plot_type = avg_success_rates_series
        y_label = "Actual Success Rate"
        file_name = "actual"
    if type == P:
        plot_type = avg_stopped_repairs_series
        y_label = "Perceived Success Rate"
        file_name = "perceived"
    if type == O:
        plot_type = avg_overfitting_series
        y_label = "Overfitting Rate"
        file_name = "overfitting"

    labels = [x.label for x in eval_datas]

    bars = plt.bar(
        range(1, num_groups + 1),
        plot_type,
        bar_width,
        label=labels,
        color=base_colors,
        edgecolor='black',
        alpha=0.9,
        zorder=2,
    )

    for bar in bars:
        height = bar.get_height()
        plt.text(
            bar.get_x() + bar.get_width()/2,  # X position (center of the bar)
            height + 0.02,                    # Y position (slightly above the bar)
            f'{height:.2f}',                   # Text format (2 decimal places)
            ha='center', va='bottom', fontsize=10
        )

    plt.ylim(0, 1.5)
    plt.yticks(np.arange(0, 1.1, 0.2))  # Ticks only from 0 to 1
    plt.gca().set_yticklabels([f"{tick :.1f}" if tick <= 1 else "" for tick in plt.gca().get_yticks()])  # Hide labels > 1
    plt.grid(axis='y', linestyle='-', alpha=0.6, zorder=0)

    plt.ylabel(y_label, fontsize=12)
    plt.yticks(fontsize=10)
    plt.legend(
        title="Modifiers",
        fontsize=10,
        title_fontsize=12,
        loc="upper left",
        bbox_to_anchor=(0, 1),  # Align top-left of legend to top-left of the plot
        ncol=1  # Keep items in a single column
        )
    plt.grid(axis='y', linestyle='-', alpha=0.6, zorder = 0)
    plt.tight_layout()
    path = Path(f"plots/total_averages")
    path.mkdir(exist_ok=True, parents=True)
    plt.savefig(path / f"{file_name}_{subject}{bug_id}_mod{len(eval_datas)}.png", dpi=300)
    #plt.show()


def create_time_graph(subject, bug_id, eval_data: EvalData):

    all_repair_durations = []
    all_evaluation_durations = []

    for frame in eval_data.frames:
        filtered_rows = frame[(frame['subject'] == subject) & (frame['bug_id'] == bug_id)]
        all_repair_durations.append(filtered_rows['repair_duration'].values)
        all_evaluation_durations.append(filtered_rows['evaluation_duration'].values)

    repair_duration_df = pd.DataFrame(all_repair_durations).T
    evaluation_duration_df = pd.DataFrame(all_evaluation_durations).T

    avg_repair_durations = repair_duration_df.mean(axis=1)
    avg_evaluation_durations = evaluation_duration_df.mean(axis=1)

    x_labels = [
        "B1-1", "B1-10", "F5-5", "F10-10", "F30-30", "F50-50",
        "V5-5", "V10-10", "V30-30", "V50-50", "C5-5", "C10-10", "C30-30", "C50-50"
    ]

    out = ""
    for i, label in enumerate(x_labels):
        out += f" & {round(avg_repair_durations[i], 2)}"
    print(out + f" \\\\")

    out = ""
    for i, label in enumerate(x_labels):
        out += f" & {round(avg_evaluation_durations[i], 2)}"
    print(out + f" \\\\")

    plt.figure(figsize=(10, 6))
    plt.plot(avg_repair_durations, marker='o', linestyle='-', color='b', label='Average Repair Duration')
    plt.plot(avg_evaluation_durations, marker='s', linestyle='--', color='r', label='Average Evaluation Duration')
    plt.xlabel('Test cases (failing-passing)')
    plt.ylabel('Duration (seconds)')
    plt.xticks(ticks=range(len(x_labels)), labels=x_labels, rotation=45, ha='right')
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    #plt.show()
    #path = Path(f"plots/time/{eval_data.save_location}")
    #path.mkdir(exist_ok=True, parents=True)
    #plt.savefig(path / f"{subject}_{bug_id}.png", dpi=300)


A = "ACTUAL"
P = "PERCEIVED"
O = "OVERFITTING"

""" EVALS = [
    ("TopEqualRankModifier (w_mut=0.20)", "results/toy_equal_20/#/csv_files/data_#.csv", "toy_equal_20"),
] """

EVALS = [
    ("DefaultModifier (w_mut=0.06)", "results/toy_default_06/#/csv_files/data_#.csv", "toy_default_06"),
    ("DefaultModifier (w_mut=0.20)", "results/toy_default_20/#/csv_files/data_#.csv", "toy_default_20"),
    #("TopRankModifier (w_mut=0.20)", "results/toy_rank_20/#/csv_files/data_#.csv", "toy_rank_20"),
    ("TopEqualRankModifier (w_mut=0.20)", "results/toy_equal_20/#/csv_files/data_#.csv", "toy_equal_20"),
    #("WeightedTopRankModifier (w_mut=0.20)", "results/toy_weighted_20/#/csv_files/data_#.csv", "toy_weighted_20"),
    #("SigmoidModifier (w_mut=0.20)", "results/toy_sigmoid_20/#/csv_files/data_#.csv", "toy_sigmoid_20"),
]

def read_csv(seed, path):
    file = path.replace("#", str(seed))
    return pd.read_csv(file)

def collect_dataframes(label, path, save_location):
    df1 = read_csv(1714, path)
    df2 = read_csv(3948, path)
    df3 = read_csv(5233, path)
    df4 = read_csv(7906, path)
    df5 = read_csv(9312, path)
    return EvalData(label, [df1, df2, df3, df4, df5], save_location)

eval_datas = []
for eval in EVALS:
    eval_datas.append(collect_dataframes(eval[0], eval[1], eval[2]))

subject = "middle"
bug_id = 1
#create_time_graph(subject, bug_id, eval_datas[0])
create_success_graph(subject, bug_id, O, eval_datas)
#create_average_graph(subject, bug_id, A, eval_datas)

        

