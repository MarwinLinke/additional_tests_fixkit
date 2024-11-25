import shutil
import numpy as np
from pathlib import Path
from typing import Type, Dict, Any, Tuple

from tests4py.projects import Project

from data import get_evaluation_data, SUBJECT_PARAMS, VARIANTS
from pipeline import EvaluationPipeline


def evaluate(project_name: str, bug_id: int, variant: str, iterations: int, number_test_cases: int, negated_formula: bool):
    
    if Path("rep").exists():
        shutil.rmtree("rep")

    setup = get_evaluation_data("GENPROG", project_name, bug_id)
    approach, parameters, subject, benchmark_program = setup
    
    num_tests4py_tests = 0
    num_avicenna_tests = 0

    match variant:
        case "BASELINE":
            use_avicenna_fault_localization = False
            use_avicenna_validation = False
            num_tests4py_tests = number_test_cases
        case "FAULT_LOCALIZATION":
            use_avicenna_fault_localization = True
            use_avicenna_validation = False
            num_tests4py_tests = 1
            num_avicenna_tests = number_test_cases
        case "VALIDATION":
            use_avicenna_fault_localization = False
            use_avicenna_validation = True
            num_tests4py_tests = 1
            num_avicenna_tests = number_test_cases
        case "COMPLETE":
            use_avicenna_fault_localization = True
            use_avicenna_validation = True
            num_avicenna_tests = number_test_cases
        case _:
            raise ValueError("Variant not allowed.")      

    use_cached_tests = True
    avicenna_iterations = 10
    use_parallel_engine = False
    
    pipeline = EvaluationPipeline(
        approach=approach,
        parameters=parameters,
        subject=subject,
        benchmark_program=benchmark_program,
        fixkit_iterations=iterations,
        num_tests4py_tests=num_tests4py_tests,
        use_avicenna_fault_localization=use_avicenna_fault_localization,
        use_avicenna_validation=use_avicenna_validation,
        avicenna_iterations=avicenna_iterations,
        num_avicenna_tests=num_avicenna_tests,
        use_cached_tests=use_cached_tests,
        use_negated_formula=negated_formula,
        use_parallel_engine=use_parallel_engine
    )

    pipeline.run()    
    pipeline.write_to_csv("tmp/data.csv")

    variant = "C" if use_avicenna_fault_localization and use_avicenna_validation else "F" if use_avicenna_fault_localization else "V" if use_avicenna_validation else "B"
    tests_identifier_avicenna = f"A{num_avicenna_tests}" if variant != "B" else ""
    tests_identifier_tests4py = f"T{num_tests4py_tests}" if variant != "C" else ""
    engine_identifier = "P" if use_parallel_engine else "S"
    file_name = f"{approach.__name__}_{subject.project_name}{subject.bug_id}_I{iterations}-{variant}-{tests_identifier_avicenna}{tests_identifier_tests4py}-{engine_identifier}"

    out = Path("out", f"{file_name}.txt")
    with open(out, "w") as f:
        f.write(pipeline.output)


def evaluate_all():
    EvaluationPipeline.write_csv_header("tmp/data.csv")

    subjects_to_evaluate = ["MIDDLE_1"] # SUBJECT_PARAMS.keys() for all

    for subject in subjects_to_evaluate:
        param = SUBJECT_PARAMS[subject]
        project_name = param["SUBJECT"]
        bug_id = param["BUG_ID"]
        negated_formula = param["NEGATED_FORMULA"]
        for variant in VARIANTS.items():
            variant_name, test_cases_version = variant
            for iteration in param["ITERATIONS"]:
                for test_cases in param[test_cases_version]:
                    
                    print(f"{project_name}, {bug_id}, {variant_name}, {iteration}, {test_cases}, {negated_formula}")
                    evaluate(project_name, bug_id, variant_name, iteration, test_cases, negated_formula)

if __name__ == "__main__":
    evaluate_all()
