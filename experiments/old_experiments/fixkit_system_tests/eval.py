import random
import time
from pathlib import Path
from typing import Type, Dict, Any, Tuple

import numpy as np
import tests4py.api as t4p
from tests4py.api.test import TestResult
from tests4py.api.report import TestReport
from tests4py.projects import Project

from fixkit.constants import DEFAULT_EXCLUDES
from fixkit.fitness.engine import Tests4PyEngine, Tests4PySystemTestEngine, Tests4PySystemTestSequentialEngine
from fixkit.repair.patch import get_patch
from fixkit.fitness.metric import AbsoluteFitness
from fixkit.localization.t4p import *
from fixkit.repair import GeneticRepair
from fixkit.repair.pygenprog import PyGenProg
from fixkit.repair.pykali import PyKali
from fixkit.repair.pymutrepair import PyMutRepair
from fixkit.repair.pycardumen import PyCardumen
from fixkit.repair.pyae import PyAE
from fixkit.candidate import GeneticCandidate

from debugging_framework.benchmark.repository import BenchmarkRepository
from debugging_benchmark.calculator.calculator import CalculatorBenchmarkRepository
from debugging_benchmark.middle.middle import MiddleBenchmarkRepository
from debugging_benchmark.markup.markup import MarkupBenchmarkRepository
from debugging_benchmark.expression.expression import ExpressionBenchmarkRepository
from debugging_benchmark.tests4py_benchmark.repository import PysnooperBenchmarkRepository

from confusion_matrix import RepairEvaluationMatrix
from test_case_fuzzer import Tests4PyFuzzer


random.seed(0)
np.random.seed(0)

APPROACHES = {
    "GENPROG": (
        PyGenProg,
        {
            "population_size": 40,
            "w_mut": 0.06,
            "workers": 32,
        },
    ),
    "KALI": (
        PyKali,
        {
            "w_mut": 0.06,
            "workers": 32,
        },
    ),
    "MUTREPAIR": (
        PyMutRepair,
        {

            "w_mut": 0.06,
            "workers": 32,
        },
    ),
    "DEEPREPAIR": (None, {}),
    "CARDUMEN": (
        PyCardumen,
        {
            "population_size": 40,
            "w_mut": 0.06,
            "workers": 32,
        },
    ),
    "AE": (PyAE, {"k": 1}),
    "SPR": (None, {}),
}

SUBJECTS = {
    "MIDDLE": ({
        1: t4p.middle_1,
        2: t4p.middle_2,
    }, 
    MiddleBenchmarkRepository()),

    "MARKUP": ({
        1: t4p.markup_1,
        2: t4p.markup_2,
    }, 
    MarkupBenchmarkRepository()),

    "EXPRESSION": ({
        1: t4p.expression_1,
    }, 
    ExpressionBenchmarkRepository()),

    "CALCULATOR": ({
        1: t4p.calculator_1,
    }, 
    CalculatorBenchmarkRepository()),

    "PYSNOOPER": ({
        1: t4p.pysnooper_1,
        2: t4p.pysnooper_2,
        3: t4p.pysnooper_3,
    }, 
    PysnooperBenchmarkRepository())
}

def almost_equal(value, target, delta=0.0001):
    return abs(value - target) < delta


def evaluate(
    approach: Type[GeneticRepair], 
    subject: Project,
    harness_id: int,
    harness: BenchmarkRepository, 
    parameters: Dict[str, Any], 
    iterations: int,
    count_test_cases: int
):
    report = t4p.checkout(subject)
    if report.raised:
        raise report.raised
    start = time.time()

    passing_test_cases = [
        os.path.abspath(
            os.path.join(
                "tmp",
                str(subject.get_identifier()),
                "tests4py_systemtest_diversity",
                f"passing_test_diversity_{i}",
            )
        ) for i in range(count_test_cases)
    ]

    failing_test_cases = [
        os.path.abspath(
            os.path.join(
                "tmp",
                str(subject.get_identifier()),
                "tests4py_systemtest_diversity",
                f"failing_test_diversity_{i}",
            )
        ) for i in range(count_test_cases)
    ]

    test_cases = passing_test_cases + failing_test_cases
    
    print(test_cases)

    approach = approach.from_source(
        src=Path("tmp", subject.get_identifier()),
        excludes=DEFAULT_EXCLUDES,
        localization=Tests4PySystemtestsLocalization(
            src=Path("tmp", subject.get_identifier()),
            events=["line"],
            predicates=["line"],
            metric="Ochiai",
            out="rep",
            tests = test_cases

        ),
        out="rep",
        is_t4p=True,
        max_generations=iterations,
        is_system_test = True,
        system_tests = test_cases,
        **parameters,
    )
    patches: List[GeneticCandidate] = approach.repair()
    duration_fixkit = time.time() - start

    print("Starting evaluation engine.")

    out = Path("evaluation_test_cases")
    fuzzer = Tests4PyFuzzer(harness, harness_id, out)
    fuzzer.generate_test_cases(50, 50)
    evaluation_passing = Tests4PyFuzzer.load_passing_test_paths(out)
    evaluation_failing = Tests4PyFuzzer.load_failing_test_paths(out)
    evaluation_test_cases = evaluation_passing + evaluation_failing

    found = False
    best_fitness = 0.0
    matrices = []

    engine = Tests4PySystemTestSequentialEngine(AbsoluteFitness(set(), set()), evaluation_test_cases, out="rep")
    engine.evaluate(patches)
    for patch in patches:

        if patch.tests4py_report:
            passing: List[str] = []
            failing: List[str] = []
            results: Dict[str, Tuple[TestResult, str]] = patch.tests4py_report.results

            for test, (result, _) in results.items():
                if result == TestResult.PASSING:
                    passing.append(test)
                elif result == TestResult.FAILING:
                    failing.append(test)

            matrix = RepairEvaluationMatrix(evaluation_passing, evaluation_failing, passing, failing)
            matrices.append(matrix)
            print(str(matrix))
        else:
            matrices.append(None)

        if almost_equal(patch.fitness, 1):
            best_fitness = max(best_fitness, patch.fitness)
            print(get_patch(patch))
            found = True
            break

    duration_eval = time.time() - start - duration_fixkit

    return patches, found, best_fitness, duration_fixkit, duration_eval, matrices


def format_output(approach, subject, patches, found, duration_fixkit, duration_eval, best_fitness, matrices) -> str:
    output: str = f"APPROACH: {approach.__name__}\n"
    output += f"SUBJECT: {subject.get_identifier()}\n"
    output += f"The repair ran for {"{:.4f}".format(duration_fixkit)} seconds.\n"
    output += f"The evaluation took {"{:.4f}".format(duration_eval)} seconds.\n"
    output += f"Was a valid patch found: {found}\n"
    output += f"BEST FITNESS: {best_fitness}\n"
    output += f"\nPATCHES:\n"

    matrix_iter = iter(matrices)
    for patch in patches:
        output += str(patch)
        matrix = str(next(matrix_iter)) or "Report wasn't valid."
        output += matrix
        fix = get_patch(patch)
        if fix == "":
            output += "\nPatch couldn't be printed."
        else:
            output += f"\n{fix}\n"

    return output


def main():
    approach = APPROACHES["GENPROG"]
    approach, parameters = approach
    subject_harness = SUBJECTS["CALCULATOR"]
    bug_id = 1
    harness_id = 0
    subject_dict, harness = subject_harness
    subject = subject_dict[bug_id]
    iterations = 10
    count_test_cases = 10

    patches, found, best_fitness, duration_fixkit, duration_eval, matrices = evaluate(
        approach=approach,
        subject=subject,
        harness=harness,
        harness_id=harness_id,
        parameters=parameters,
        iterations=iterations,
        count_test_cases=count_test_cases
        )
    
    out = Path("out", f"{approach.__name__}_{subject.get_identifier()}_iter{iterations}_tests{count_test_cases}.txt")
    with open(out, "w") as f:
        f.write(format_output(approach,subject, patches, found, duration_fixkit, duration_eval, best_fitness, matrices))

if __name__ == "__main__":
    main()
