import pandas as pd


variant_names = {"B" : "Baseline", "F" : "Fault Localization", "V" : "Validation", "C": "Complete"}

def read_csv(file):
    return pd.read_csv(file)


def create_rows(subject, bug_id, *frames):

    row_string = ""
    frame1: pd.DataFrame = frames[0]
    old_variant = "Baseline"

    filtered_frame1 = frame1[(frame1["subject"] == subject) & (frame1["bug_id"] == bug_id)]

    for idx, row in filtered_frame1.iterrows():

        variant = variant_names[row["variant"]]

        if variant != old_variant:
            old_variant = variant
            row_string += "\\midrule \n"
    
        iterations = row["iterations"]
        num_failing = row["baseline_failing"] if variant == "Baseline" else row["additional_failing"]
        num_passing = row["baseline_passing"] if variant == "Baseline" else row["additional_passing"]
        checkmark = "\\cmark" if float(row["precision"]) == 1.0 else ""
        f1_score = round(float(row["f1_score"]), 2)
        f1_score = "\\dvalue" if f1_score == 0.0 else f1_score

        row_string += f"{variant} & {iterations} & ({num_failing}, {num_passing}) & {checkmark}{f1_score} "

        for frame in frames[1:]:
            checkmark = "\\cmark" if float(frame.loc[idx, "precision"]) == 1.0 else ""
            f1_score = round(float(frame.loc[idx, "f1_score"]), 2)
            f1_score = "\\dvalue" if f1_score == 0.0 else f1_score
            row_string +=  f"& {checkmark}{f1_score} "

        row_string += "\\\\ \n"

    return row_string

df1 = read_csv("results/1714/data_seed_1714.csv")
df2 = read_csv("results/3948/data_seed_3948.csv")
df3 = read_csv("results/5233/data_seed_5233.csv")
df4 = read_csv("results/7906/data_seed_7906.csv")
df5 = read_csv("results/9312/data_seed_9312.csv")
subject = "markup"
bug_id = 2
print(create_rows(subject, bug_id, df1, df2, df3, df4, df5))
