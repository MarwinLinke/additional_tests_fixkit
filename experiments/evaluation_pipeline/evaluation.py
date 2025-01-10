import shutil
from pathlib import Path
import csv
import random
import logging
import numpy as np
import sys

from fixkit.logger import debug_logger
from data import get_evaluation_data, get_benchmark_program, get_subject_params, VARIANTS
from pipeline import EvaluationPipeline
from pre_generator import PreGenerator

OUT = "out"
REP = "rep"
TMP = "tmp"
SFL = "sflkit_events"
REPAIR_TESTS = "repair_tests"
EVAL_TESTS = "evaluation_tests"
PYCACHE = "__pycache__"

BASELINE = "BASELINE"
LOCALIZATION = "FAULT_LOCALIZATION"
VALIDATION = "VALIDATION"
COMPLETE = "COMPLETE"

def _evaluate(project_name: str, bug_id: int, variant: str, iterations: int, num_failing: int, num_passing: int, repair_tests_path: Path, evaluation_tests_path: Path, csv_name: str, seed: int):
    
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
        enhance_localization=enhance_fault_localization,
        enhance_validation=enhance_validation,
        use_parallel_engine=use_parallel_engine
    )

    pipeline.run()    
    pipeline.write_to_csv(Path(OUT) / "csv_files", csv_name)
    pipeline.write_report(Path(OUT) / "reports" / str(seed))


def _try_evaluate(project_name: str, bug_id: int, variant_name: str, iteration: int, num_failing: int, num_passing: int, repair_tests_path: Path, evaluation_tests_path: Path, csv_name: str, seed: int):
    """
    Tries to run evaluation and handles exceptions.
    """
    print(f"Evaluating {project_name}, {bug_id}, {variant_name}, {iteration}, {num_failing}, {num_passing}, {csv_name}.")
    
    try:
        if SIMULATE_EVALUATION:
            raise NotImplementedError("This is only a simulation, please disable the flag for SIMULATE_EVALUATION.")

        _evaluate(project_name, bug_id, variant_name, iteration, num_failing, num_passing, repair_tests_path, evaluation_tests_path, csv_name, seed)                                 
    
    except Exception as exception:
        with open(Path(OUT) / "csv_files" / csv_name, "a", newline="") as file:
            writer = csv.writer(file)
            data = ["GENPROG", project_name, bug_id, iteration, variant_name, num_failing, num_passing,
                type(exception).__name__, str(exception), seed, None, None, None, None, None, None, None,
                None, None, None, None ]
            writer.writerow(data)


def _generate_repair_tests(subject: str, seed: int, num_failing: int = 50, num_passing: int = 50):
    """
    Generates and saves test cases for repair.
    """
    path = Path("repair_tests")
    params = get_subject_params(subject)
    benchmark_program = get_benchmark_program(params["SUBJECT"], params["BUG_ID"])
    pre_generator = PreGenerator(subject, benchmark_program, num_failing, num_passing, path, seed)
    return pre_generator.run()


def _generate_evaluation_tests(subject: str, seed: int, num_failing: int = 50, num_passing: int = 50):
    """
    Generates and saves test cases for evaluation.
    """
    path = Path("evaluation_tests")
    params = get_subject_params(subject)
    benchmark_program = get_benchmark_program(params["SUBJECT"], params["BUG_ID"])
    pre_generator = PreGenerator(subject, benchmark_program, num_failing, num_passing, path, seed)
    return pre_generator.run()


def _evaluate_subject(subject: str, seed: int, csv_file: Path, repair_tests_path: Path, evaluation_tests_path: Path):
    """
    Evaluates specified configurations in data.py for the subject.
    """
    print(f"Starting the evaluation of {subject}.")

    params = get_subject_params(subject)
    project_name = params["SUBJECT"]
    bug_id = params["BUG_ID"]
    for iteration in params["ITERATIONS"]:
        for variant in VARIANTS.items():
            variant_name, num_tests_version = variant
            for num_tests in params[num_tests_version]:             
                num_failing, num_passing = num_tests
                _try_evaluate(project_name, bug_id, variant_name, iteration, num_failing, num_passing, repair_tests_path, evaluation_tests_path, csv_file, seed)


SEEDS = [1714, 3948, 5233, 7906, 9312]
REPAIR_TESTS_SEEDS = [959, 2655, 4916, 6114, 8452]
EVAL_TESTS_SEED = 0

SIMULATE_EVALUATION = False

def evaluate_all(index):
    """
    Evaluates all specified subjects. Generates test cases before every subject for every seed.
    Currently uses fixed seeds for test cases generation.
    """
    if index == -1:
        clean_up()
        return

    logging.getLogger("tests4py").propagate = False

    if index >= len(SEEDS) or index < 0:
        raise ValueError(f"Index for seeds must be in valid range: [0, {len(SEEDS) - 1}]")

    subjects_to_evaluate = ["PYSNOOPER_2, PYSNOOPER_3"]

    shutil.rmtree(REPAIR_TESTS, ignore_errors=True)
    shutil.rmtree(EVAL_TESTS, ignore_errors=True)

    seed = SEEDS[index]
    repair_tests_seed = REPAIR_TESTS_SEEDS[index]
    csv_file = f"data_{seed}.csv"
    EvaluationPipeline.write_csv_header(Path(OUT) / "csv_files", csv_file)
    for subject in subjects_to_evaluate:
        repair_tests_path = _generate_repair_tests(subject, repair_tests_seed)
        evaluation_tests_path = _generate_evaluation_tests(subject, EVAL_TESTS_SEED)
        _evaluate_subject(subject, seed, csv_file, repair_tests_path, evaluation_tests_path)
            

def debug_evaluation():
    
    logging.getLogger("tests4py").propagate = False
    debug_logger()

    seed_index = 0
    subject = "MIDDLE"
    bug_id = 1
    repair_tests_path = _generate_repair_tests(f"{subject}_{bug_id}", REPAIR_TESTS_SEEDS[seed_index], 10, 10)
    eval_tests_path = _generate_evaluation_tests(f"{subject}_{bug_id}", EVAL_TESTS_SEED, 10, 10)
    csv = "seed_test.csv"
    EvaluationPipeline.write_csv_header(Path(OUT) / "csv_files", csv)
    for _ in range(10):
        _evaluate(subject, bug_id, BASELINE, 10, 1, 10, repair_tests_path, eval_tests_path, csv, SEEDS[seed_index])


def clean_up():
    shutil.rmtree(REPAIR_TESTS, ignore_errors=True)
    shutil.rmtree(EVAL_TESTS, ignore_errors=True)
    shutil.rmtree(REP, ignore_errors=True)
    shutil.rmtree(TMP, ignore_errors=True)
    shutil.rmtree(SFL, ignore_errors=True)
    shutil.rmtree(PYCACHE, ignore_errors=True)


if __name__ == "__main__":
    args = sys.argv[1:]

    if len(args) == 0:
        debug_evaluation()

    for arg in args:
        evaluate_all(int(arg))
