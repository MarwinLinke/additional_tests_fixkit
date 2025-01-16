import pandas as pd
import matplotlib.pyplot as plt
from typing import List
from pathlib import Path
import seaborn as sns

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

def create_success_graph(subject: str, bug_id: int, actual: bool, eval_datas: List[EvalData]):
    x_labels = [
        "B1-1", "B1-10", "F5-5", "F10-10", "F30-30", "F50-50",
        "V5-5", "V10-10", "V30-30", "V50-50", "C5-5", "C10-10", "C30-30", "C50-50"
    ]

    num_groups = len(eval_datas)
    group_width = 0.75
    bar_width = group_width / num_groups

    avg_stopped_repairs_matrix = []
    avg_success_rates_matrix = []

    for eval_data in eval_datas:
        frames = eval_data.frames
        stopped_repairs = []
        success_rates = []
        for frame in frames:
            filtered_rows = frame[(frame['subject'] == subject) & (frame['bug_id'] == bug_id)]
            stopped_repairs.append([
                gen < 10 or (gen == 10 and score == 1.0)
                for gen, score in zip(filtered_rows["generations"].values, filtered_rows["f1_score"].values)
            ])
            success_rates.append([
                score == 1.0 for score in filtered_rows["f1_score"].values
            ])

        stopped_repairs_df = pd.DataFrame(stopped_repairs).T
        avg_stopped_repairs = stopped_repairs_df.sum(axis=1) / len(frames)
        avg_stopped_repairs_matrix.append(avg_stopped_repairs)

        success_rates_df = pd.DataFrame(success_rates).T
        avg_success_rates = success_rates_df.sum(axis=1) / len(frames)
        avg_success_rates_matrix.append(avg_success_rates)

    base_colors = sns.color_palette("YlGnBu", num_groups)

    plt.figure(figsize=(12, 6))
    x = range(len(x_labels))

    for i, (stopped_repairs, success_rate) in enumerate(zip(avg_stopped_repairs_matrix, avg_success_rates_matrix)):
        offset = (i - (num_groups - 1) / 2) * bar_width
        plt.bar(
            [pos + offset for pos in x],
            success_rate if actual else stopped_repairs,
            bar_width,
            label=f'{eval_datas[i].label}',
            color=base_colors[i],
            edgecolor='black',
            alpha=0.9,
            zorder=2,
        )

    plt.ylim(0, 1)
    plt.axvline(x = 1.5, color='grey', linewidth=0.8, linestyle='-', alpha=0.4, zorder = 0)
    plt.axvline(x = 5.5, color='grey', linewidth=0.8, linestyle='-', alpha=0.4, zorder = 0)
    plt.axvline(x = 9.5, color='grey', linewidth=0.8, linestyle='-', alpha=0.4, zorder = 0)
    plt.xlabel('Test cases (failing-passing)', fontsize=12)
    plt.ylabel(f'{"Actual" if actual else "Perceived"} Success Rate', fontsize=12)
    plt.xticks(ticks=x, labels=x_labels, rotation=45, ha='right', fontsize=10)
    plt.yticks(fontsize=10)
    plt.legend(title="Modifiers", fontsize=10, title_fontsize=12)
    plt.grid(axis='y', linestyle='-', alpha=0.6, zorder = 0)
    plt.tight_layout()
    path = Path(f"plots/success_rates")
    path.mkdir(exist_ok=True, parents=True)
    plt.savefig(path / f"{"actual" if actual else "perceived"}"
                f"_{subject}{bug_id}_mod{len(eval_datas)}.png", dpi=300)
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
    path = Path(f"plots/time/{eval_data.save_location}")
    path.mkdir(exist_ok=True, parents=True)
    plt.savefig(path / f"{subject}_{bug_id}.png", dpi=300)


EVALS = [
    ("DefaultModifier (w_mut=0.06)", "results/toy_default_06/#/csv_files/data_#.csv", "toy_default_06"),
    ("DefaultModifier (w_mut=0.20)", "results/toy_default_20/#/csv_files/data_#.csv", "toy_default_20"),
    ("TopEqualRankModifier (w_mut=0.20)", "results/toy_equal_20/#/csv_files/data_#.csv", "toy_equal_20"),
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

subject = "calculator"
bug_id = 1
#create_time_graph(subject, bug_id, eval_datas[0])
create_success_graph(subject, bug_id, False, eval_datas)

        

