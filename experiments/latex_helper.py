import pandas as pd


variant_names = {"B" : "Baseline", "F" : "Localization", "V" : "Validation", "C": "Complete"}

def read_csv(seed):
    file = PATH.replace("#", str(seed))
    return pd.read_csv(file)

def create_header(caption, *frames):
    header = """
\\begin{table}[]
    \\centering
    \\caption{#}
    \\vspace{5mm}
    \\scriptsize   
    \\begin{tabular}{lc|rrrrr}
        \\toprule
        \\multirow{2}{*}{Variant} & \\multirow{2}{*}{Test cases} & \\multicolumn{5}{c}{\\textsc{Seeds}} \\\\
        &  & \\multicolumn{1}{c}{1714} & \\multicolumn{1}{c}{3948} & \\multicolumn{1}{c}{5233} & \\multicolumn{1}{c}{7906} & \\multicolumn{1}{c}{9312} \\\\
"""

    return header.replace("#", caption)


def create_rows(subject, bug_id, indentation, *frames):

    row_string = ""
    frame1: pd.DataFrame = frames[0]
    old_variant = ""

    filtered_frame1 = frame1[(frame1["subject"] == subject) & (frame1["bug_id"] == bug_id)]

    for idx, row in filtered_frame1.iterrows():

        indentation_space = int(indentation) * "    " 
        variant = variant_names[row["variant"]]
        variant_text = ""

        if variant != old_variant: 
            variant_text = "\\multirow{2}{*}{#}" if variant == "Baseline" else "\\multirow{4}{*}{#}"
            variant_text = variant_text.replace("#", variant) 
            old_variant = variant
            row_string += f"{indentation_space}\\midrule \n"
    
        iterations = row["iterations"]
        num_failing = row["baseline_failing"] if variant == "Baseline" else row["additional_failing"]
        num_passing = row["baseline_passing"] if variant == "Baseline" else row["additional_passing"]
        
        #checkmark = "\\cmark" if float(row["precision"]) == 1.0 else ""
        f1_score = round(float(row["f1_score"]), 2)
        f1_score_text = "\\dvalue" if f1_score == 0.0 else f"{f1_score:.2f}"
        gen = int(row["generations"])
        generations = f"({gen}) " if gen != iterations or f1_score == 1.0 else ""
       
        row_string += f"{indentation_space}{variant_text} & ({num_failing}, {num_passing}) & {generations}{f1_score_text} "

        for frame in frames[1:]:
            # checkmark = "\\cmark" if float(frame.loc[idx, "precision"]) == 1.0 else ""
            f1_score = round(float(frame.loc[idx, "f1_score"]), 2)
            f1_score_text = "\\dvalue" if f1_score == 0.0 else f"{f1_score:.2f}"
            gen = int(frame.loc[idx, "generations"])
            generations = f"({gen}) " if gen != iterations or f1_score == 1.0 else ""
            row_string +=  f"& {generations}{f1_score_text} "

        row_string += "\\\\ \n"

    return row_string.strip("\n")

def create_tail(label):
    tail = """
        \\bottomrule
    \\end{tabular}
    \\label{#}
\\end{table}
"""
    return tail.replace("#", label)

def create_table(subject, bug_id, caption, label, indentation, df1, df2, df3, df4, df5):
    return create_header(caption) + create_rows(subject, bug_id, 2, df1, df2, df3, df4, df5) + create_tail(label)

PATH = "results/toy_equal_20/#/csv_files/data_#.csv"
CAPTION = "Results of #."

df1 = read_csv(1714)
df2 = read_csv(3948)
df3 = read_csv(5233)
df4 = read_csv(7906)
df5 = read_csv(9312)
subject = "middle"
bug_id = 2
caption = CAPTION.replace("#", "\\middletwo")
label = f"tab:results:{subject}{bug_id}:default"
print(create_table(subject, bug_id, caption, label, 2, df1, df2, df3, df4, df5))
