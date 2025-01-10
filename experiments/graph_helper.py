import pandas as pd
import matplotlib.pyplot as plt

def create_graph(subject, bug_id, *frames):
    all_repair_durations = []
    all_evaluation_durations = []
    success_rate = []

    for frame in frames:
        filtered_rows = frame[(frame['subject'] == subject) & (frame['bug_id'] == bug_id)]
        all_repair_durations.append(filtered_rows['repair_duration'].values)
        all_evaluation_durations.append(filtered_rows['evaluation_duration'].values)
        success_rate.append([value == 1.0 for value in filtered_rows["f1_score"].values])

    repair_duration_df = pd.DataFrame(all_repair_durations).T
    evaluation_duration_df = pd.DataFrame(all_evaluation_durations).T
    success_rate_df = pd.DataFrame(success_rate).T

    avg_repair_durations = repair_duration_df.mean(axis=1)
    avg_evaluation_durations = evaluation_duration_df.mean(axis=1)
    avg_success_rate = success_rate_df.sum(axis=1) / 5

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
    plt.show()
    #plt.savefig(f"plots/time_{subject}_{bug_id}.png", dpi=300)

    plt.figure(figsize=(10, 6))
    plt.bar(x_labels, avg_success_rate, color='g', alpha=0.7, label='Average Success Rate')
    plt.xlabel('Test cases (failing-passing)')
    plt.ylabel('Success Rate')
    plt.xticks(rotation=45, ha='right')
    plt.legend()
    plt.tight_layout()
    plt.show()
    #plt.savefig(f"plots/succes_rate_{subject}_{bug_id}.png", dpi=300)

PATH = "results/results_v3/#/csv_files/data_#.csv"

def read_csv(seed):
    file = PATH.replace("#", str(seed))
    return pd.read_csv(file)

df1 = read_csv(1714)
df2 = read_csv(3948)
df3 = read_csv(5233)
df4 = read_csv(7906)
df5 = read_csv(9312)

subject = "middle"
bug_id = 2
create_graph(subject, bug_id, df1, df2, df3, df4, df5)
