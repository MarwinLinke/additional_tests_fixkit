import shutil
from pathlib import Path
import csv
import random
import numpy as np

from data import get_evaluation_data, SUBJECT_PARAMS, VARIANTS
from pipeline import EvaluationPipeline, TestGenerationType

def evaluate(project_name: str, bug_id: int, variant: str, iterations: int, num_faling: int, num_passing: int, lock_evaluation_tests: bool, csv: str, seed: int):
    
    if Path("rep").exists():
        shutil.rmtree("rep")

    setup = get_evaluation_data("GENPROG", project_name, bug_id)
    approach, parameters, subject, benchmark_program = setup
    
    num_baseline_failing = 0
    num_baseline_passing = 0
    num_additional_failing = 0
    num_additional_passing = 0

    match variant:
        case "BASELINE":
            enhance_fault_localization = False
            enhance_validation = False
            num_baseline_failing = num_faling
            num_baseline_passing = num_passing
        case "FAULT_LOCALIZATION":
            enhance_fault_localization = True
            enhance_validation = False
            num_baseline_failing = 1
            num_baseline_passing = 10
            num_additional_failing = num_faling
            num_additional_passing = num_passing
        case "VALIDATION":
            enhance_fault_localization = False
            enhance_validation = True
            num_baseline_failing = 1
            num_baseline_passing = 10
            num_additional_failing = num_faling
            num_additional_passing = num_passing
        case "COMPLETE":
            enhance_fault_localization = True
            enhance_validation = True
            num_additional_failing = num_faling
            num_additional_passing = num_passing
        case _:
            raise ValueError("Variant not allowed.")      

    use_cached_tests = True
    use_parallel_engine = False
    
    random.seed(seed)
    np.random.seed(seed)

    pipeline = EvaluationPipeline(
        approach=approach,
        parameters=parameters,
        subject=subject,
        benchmark_program=benchmark_program,
        repair_iterations=iterations,
        seed=seed,
        test_generation_type=TestGenerationType.GRAMMAR_FUZZER,
        num_baseline_failing=num_baseline_failing,
        num_baseline_passing=num_baseline_passing,
        num_additional_failing=num_additional_failing,
        num_additional_passing=num_additional_passing,
        num_evaluation_failing=50,
        num_evaluation_passing=50,
        lock_evaluation_tests=lock_evaluation_tests,
        enhance_fault_localization=enhance_fault_localization,
        enhance_validation=enhance_validation,
        use_cached_tests=use_cached_tests,
        use_parallel_engine=use_parallel_engine
    )

    pipeline.run()    
    pipeline.write_to_csv(csv)

    variant = "C" if enhance_fault_localization and enhance_validation else "F" if enhance_fault_localization else "V" if enhance_validation else "B"
    tests_identifier_avicenna = f"A{num_additional_failing}" if variant != "B" else ""
    tests_identifier_tests4py = f"T{num_baseline_failing}" if variant != "C" else ""
    engine_identifier = "P" if use_parallel_engine else "S"
    file_name = f"{approach.__name__}_{subject.project_name}{subject.bug_id}_I{iterations}-{variant}-{tests_identifier_avicenna}{tests_identifier_tests4py}-{engine_identifier}"

    out = Path("out", f"{file_name}.txt")
    with open(out, "w") as f:
        f.write(pipeline.output)


def evaluate_all():
    """
    Evaluates all specified subjects.
    """
    seeds = [1662, 2427, 7953, 2495, 5180]
    subjects_to_evaluate = ["MIDDLE_1", "MIDDLE_2", "CALCULATOR_1", "EXPRESSION_1", "MARKUP_1", "MARKUP_2"]
    # subjects_to_evaluate = SUBJECT_PARAMS.keys()

    for seed in seeds:
        csv_file = f"tmp/data_seed_{seed}.csv"
        EvaluationPipeline.write_csv_header(csv_file)
        for subject in subjects_to_evaluate:
            param = SUBJECT_PARAMS[subject]
            project_name = param["SUBJECT"]
            bug_id = param["BUG_ID"]
            lock_evaluation_tests = False       
            for variant in VARIANTS.items():
                variant_name, num_tests_version = variant
                for iteration in param["ITERATIONS"]:
                    for num_tests in param[num_tests_version]:             
                        num_failing, num_passing = num_tests
                        try:
                            print(f"{project_name}, {bug_id}, {variant_name}, {iteration}, {num_failing}, {num_passing}, {lock_evaluation_tests}, {csv_file}")
                            evaluate(project_name, bug_id, variant_name, iteration, num_failing, num_passing, lock_evaluation_tests, csv_file, seed)                                 
                        except Exception as exception:
                            with open(csv_file, "a", newline="") as file:
                                writer = csv.writer(file)
                                data = ["GENPROG", project_name, bug_id, iteration, variant_name, num_failing, num_passing,
                                    type(exception).__name__, str(exception), seed, None, None, None, None, None, None, None,
                                    None, None, None, None ]
                                writer.writerow(data)
                        lock_evaluation_tests = True

def debug_evaluation():
    EvaluationPipeline.write_csv_header("tmp/middle_seed_1.csv")
    evaluate("MIDDLE", 2, "BASELINE", 1, 1, 10, False, "tmp/middle_seed_1.csv", 0)
    evaluate("MIDDLE", 2, "BASELINE", 1, 1, 10, True, "tmp/middle_seed_1.csv", 0)
    evaluate("CALCULATOR", 1, "BASELINE", 1, 1, 10, False, "tmp/middle_seed_1.csv", 0)


if __name__ == "__main__":
    evaluate_all()
