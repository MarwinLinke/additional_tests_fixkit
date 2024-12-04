import shutil
from pathlib import Path
import csv
import random
import logging
import numpy as np

from data import get_evaluation_data, get_benchmark_program, SUBJECT_PARAMS, VARIANTS
from pipeline import EvaluationPipeline
from pre_generator import PreGenerator

REP = "rep"
TMP = "tmp"
SFL = "sflkit_events"
REPAIR_TESTS = "repair_tests"
EVAL_TESTS = "evaluation_tests"

def _evaluate(project_name: str, bug_id: int, variant: str, iterations: int, num_failing: int, num_passing: int, repair_tests_path: Path, evaluation_tests_path: Path, csv: str, seed: int):
    
    random.seed(seed)
    np.random.seed(seed)

    shutil.rmtree(REP, ignore_errors=True)
    shutil.rmtree(TMP, ignore_errors=True)
    shutil.rmtree(SFL, ignore_errors=True)

    setup = get_evaluation_data("GENPROG", project_name, bug_id)
    approach, parameters, subject = setup
    
    num_baseline_failing = 0
    num_baseline_passing = 0
    num_additional_failing = 0
    num_additional_passing = 0

    match variant:
        case "BASELINE":
            enhance_fault_localization = False
            enhance_validation = False
            num_baseline_failing = num_failing
            num_baseline_passing = num_passing
        case "FAULT_LOCALIZATION":
            enhance_fault_localization = True
            enhance_validation = False
            num_baseline_failing = 1
            num_baseline_passing = 10
            num_additional_failing = num_failing
            num_additional_passing = num_passing
        case "VALIDATION":
            enhance_fault_localization = False
            enhance_validation = True
            num_baseline_failing = 1
            num_baseline_passing = 10
            num_additional_failing = num_failing
            num_additional_passing = num_passing
        case "COMPLETE":
            enhance_fault_localization = True
            enhance_validation = True
            num_additional_failing = num_failing
            num_additional_passing = num_passing
        case _:
            raise ValueError("Variant not allowed.")      

    use_parallel_engine = False

    pipeline = EvaluationPipeline(
        approach=approach,
        parameters=parameters,
        subject=subject,
        repair_iterations=iterations,
        seed=seed,
        repair_tests_path=repair_tests_path,
        evaluation_tests_path=evaluation_tests_path,
        num_baseline_failing=num_baseline_failing,
        num_baseline_passing=num_baseline_passing,
        num_additional_failing=num_additional_failing,
        num_additional_passing=num_additional_passing,
        num_evaluation_failing=50,
        num_evaluation_passing=50,
        enhance_fault_localization=enhance_fault_localization,
        enhance_validation=enhance_validation,
        use_parallel_engine=use_parallel_engine
    )

    pipeline.run()    
    pipeline.write_to_csv(csv)
    pipeline.write_report(Path("out") / "reports")


def _try_evaluate(project_name: str, bug_id: int, variant_name: str, iteration: int, num_failing: int, num_passing: int, repair_tests_path: Path, evaluation_tests_path: Path, csv_file: str, seed: int):
    """
    Tries to run evaluation and handles exceptions.
    """
    print(f"Evaluating {project_name}, {bug_id}, {variant_name}, {iteration}, {num_failing}, {num_passing}, {csv_file}.")
    
    try:
        if SIMULATE_EVALUATION:
            raise NotImplementedError("This is only a simulation, please disable the flag for SIMULATE_EVALUATION.")

        _evaluate(project_name, bug_id, variant_name, iteration, num_failing, num_passing, repair_tests_path, evaluation_tests_path, csv_file, seed)                                 
    
    except Exception as exception:
        with open(csv_file, "a", newline="") as file:
            writer = csv.writer(file)
            data = ["GENPROG", project_name, bug_id, iteration, variant_name, num_failing, num_passing,
                type(exception).__name__, str(exception), seed, None, None, None, None, None, None, None,
                None, None, None, None ]
            writer.writerow(data)


def _generate_repair_tests(subject: str, seed: int):
    """
    Generates and saves test cases for repair.
    """
    path = Path("repair_tests")
    benchmark_program = get_benchmark_program(SUBJECT_PARAMS[subject]["SUBJECT"], SUBJECT_PARAMS[subject]["BUG_ID"])
    pre_generator = PreGenerator(subject, benchmark_program, 50, 50, path, seed)
    return pre_generator.run()

def _generate_evaluation_tests(subject: str, seed: int):
    """
    Generates and saves test cases for evaluation.
    """
    path = Path("evaluation_tests")
    benchmark_program = get_benchmark_program(SUBJECT_PARAMS[subject]["SUBJECT"], SUBJECT_PARAMS[subject]["BUG_ID"])
    pre_generator = PreGenerator(subject, benchmark_program, 50, 50, path, seed)
    return pre_generator.run()

def _evaluate_subject(subject: str, seed: int, csv_file: Path, repair_tests_path: Path, evaluation_tests_path: Path):
    """
    Evaluates 26 combinations of parameters for the subject.
    """
    print(f"Starting the evaluation of {subject}.")

    param = SUBJECT_PARAMS[subject]
    project_name = param["SUBJECT"]
    bug_id = param["BUG_ID"]
    for variant in VARIANTS.items():
        variant_name, num_tests_version = variant
        for iteration in param["ITERATIONS"]:
            for num_tests in param[num_tests_version]:             
                num_failing, num_passing = num_tests
                _try_evaluate(project_name, bug_id, variant_name, iteration, num_failing, num_passing, repair_tests_path, evaluation_tests_path, csv_file, seed)


SEEDS = [1662, 2427, 7953, 2495, 5180]
REPAIR_TESTS_SEED = 0
EVAL_TESTS_SEED = 1

SIMULATE_EVALUATION = False

def evaluate_all():
    """
    Evaluates all specified subjects. Generates test cases before every subject for every seed.
    Currently uses fixed seeds for test cases generation.
    """
    seeds = [SEEDS[0]]
    subjects_to_evaluate = ["MIDDLE_1", "MIDDLE_2", "CALCULATOR_1", "EXPRESSION_1", "MARKUP_1", "MARKUP_2"]
    shutil.rmtree(REPAIR_TESTS, ignore_errors=True)
    shutil.rmtree(EVAL_TESTS, ignore_errors=True)

    for seed in seeds:
        csv_file = Path("out") / "csv_files" / f"data_seed_{seed}.csv"
        EvaluationPipeline.write_csv_header(csv_file)
        for subject in subjects_to_evaluate:
            repair_tests_path = _generate_repair_tests(subject, REPAIR_TESTS_SEED)
            evaluation_tests_path = _generate_evaluation_tests(subject, EVAL_TESTS_SEED)
            _evaluate_subject(subject, seed, csv_file, repair_tests_path, evaluation_tests_path)
            

def debug_evaluation():
    
    logging.getLogger("tests4py").propagate = False

    subject = "MIDDLE_1"
    repair_tests_path = _generate_repair_tests(subject, REPAIR_TESTS_SEED)
    eval_tests_path = _generate_evaluation_tests(subject, EVAL_TESTS_SEED)
    csv = "out/csv_files/seed_test.csv"
    EvaluationPipeline.write_csv_header(csv)
    for _ in range(10):
        # This generates a different number of patches despite having the same seed and same test cases! Why?
        # Also the number of patches is similar in this loop (not the same), 
        # but differs more in different executions of the python file.
        _evaluate("MIDDLE", 1, "BASELINE", 1, 1, 10, repair_tests_path, eval_tests_path, csv, SEEDS[0])

if __name__ == "__main__":
    evaluate_all()
