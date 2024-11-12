import random
import time
from pathlib import Path
from typing import Type, Dict, Any
import logging
import numpy as np
import tests4py.api as t4p
from tests4py.projects import Project

from fixkit.constants import DEFAULT_EXCLUDES
from fixkit.fitness.engine import Tests4PyEngine, Tests4PySystemTestEngine
from fixkit.fitness.metric import AbsoluteFitness
from fixkit.repair.patch import get_patch
from fixkit.localization.t4p import *
from fixkit.repair import GeneticRepair
from fixkit.repair.pygenprog import PyGenProg
from fixkit.repair.pykali import PyKali
from fixkit.repair.pymutrepair import PyMutRepair
from fixkit.repair.pycardumen import PyCardumen
from fixkit.repair.pyae import PyAE
from fixkit.test_generation.test_generator import AvicennaTestGenerator, TestGenerator

from debugging_framework.benchmark.repository import BenchmarkRepository
from debugging_benchmark.calculator.calculator import CalculatorBenchmarkRepository
from debugging_benchmark.middle.middle import MiddleBenchmarkRepository
from debugging_benchmark.markup.markup import MarkupBenchmarkRepository
from debugging_benchmark.expression.expression import ExpressionBenchmarkRepository
from debugging_benchmark.tests4py_benchmark.repository import PysnooperBenchmarkRepository


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
    iterations: int
):
    
    #logger = logging.getLogger('tests4py')
    #logger.propagate = False
    start = time.time()
    
    default_param = {
        "max_iterations": 5,
        "overwrite": True
    }

    report = t4p.checkout(subject)
    if report.raised:
        raise report.raised

    harness_program = harness.build()[harness_id]
    param = harness_program.to_dict()
    param.update(default_param)

    dir = os.path.abspath(Path("tmp") / subject.get_identifier())
    generator = AvicennaTestGenerator(out=dir, saving_method="files", **param)
    generator.run()
    generator.generate_more_inputs(100)

    failing_test_paths: List[os.PathLike] = TestGenerator.load_failing_test_paths(generator.out)
    passing_test_paths: List[os.PathLike] = TestGenerator.load_passing_test_paths(generator.out)

    test_cases = failing_test_paths + passing_test_paths

    # MANUAL ITERATION THROUGH AVICENNAS TEST CASES
    #
    #test_cases = [
    #    os.path.abspath(
    #        os.path.join(
    #            "tmp",
    #            str(subject.get_identifier()),
    #            "avicenna_test_cases",
    #            f"{status}_test_{i}",
    #        )
    #    ) for status in ["passing", "failing"] for i in range(10)
    #]  

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
    patches = approach.repair()
    duration = time.time() - start

    print("Starting evaluation engine.")

    found = False
    best_fitness = 0.0

    engine = Tests4PySystemTestEngine(AbsoluteFitness(set(), set()), test_cases, workers=32, out="rep")
    engine.evaluate(patches)
    for patch in patches:
        if almost_equal(patch.fitness, 1):
            best_fitness = max(best_fitness, patch.fitness)
            print(get_patch(patch))
            found = True
            break

    return patches, found, duration, best_fitness


def main():
    approach = APPROACHES["GENPROG"]
    approach, parameters = approach
    subject_harness = SUBJECTS["CALCULATOR"]
    bug_id = 1
    harness_id = 0
    subject_dict, harness = subject_harness
    subject = subject_dict[bug_id]
    iterations = 2
    pacthes, found, duration, best_fitness = evaluate(approach, subject, harness_id, harness, parameters, iterations)
    with open(f"{approach.__name__}_{subject.get_identifier()}.txt", "w") as f:
        f.write(  
f"""APPROACH: {approach.__name__},
SUBJECT: {subject.get_identifier()},
PATCH FOUND: {found},
DURATION: {duration},
BEST FITNESS: {best_fitness},
PATCHES: {pacthes}
"""
        )

if __name__ == "__main__":
    main()
