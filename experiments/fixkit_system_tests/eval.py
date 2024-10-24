import random
import time
from pathlib import Path
from typing import Type, Dict, Any

import numpy as np
import tests4py.api as t4p
from tests4py.projects import Project

from fixkit.constants import DEFAULT_EXCLUDES
from fixkit.fitness.engine import Tests4PyEngine
from fixkit.fitness.metric import AbsoluteFitness
from fixkit.localization.t4p import *
from fixkit.repair import GeneticRepair
from fixkit.repair.pygenprog import PyGenProg
from fixkit.repair.pykali import PyKali
from fixkit.repair.pymutrepair import PyMutRepair
from fixkit.repair.pycardumen import PyCardumen
from fixkit.repair.pyae import PyAE


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
    "MIDDLE": {
        1: t4p.middle_1,
        2: t4p.middle_2,
    },
    "MARKUP": {
        1: t4p.markup_1,
        2: t4p.markup_2,
    },
    "EXPRESSION": {
        1: t4p.expression_1,
    },
    "CALCULATOR": {
        1: t4p.calculator_1,
    },
    "PYSNOOPER": {
        1: t4p.pysnooper_1,
        2: t4p.pysnooper_2,
        3: t4p.pysnooper_3,
    }
}


def almost_equal(value, target, delta=0.0001):
    return abs(value - target) < delta


def evaluate(
    approach: Type[GeneticRepair], subject: Project, parameters: Dict[str, Any], iterations: int
):
    report = t4p.checkout(subject)
    if report.raised:
        raise report.raised
    start = time.time()

    test_cases = [
            os.path.abspath(
                os.path.join(
                    "tmp",
                    str(subject.get_identifier()),
                    "tests4py_systemtest_diversity",
                    f"{passing}_test_diversity_{i}",
                )
            ) for passing, i in zip(["passing", "failing"], range(10))
        ]
    
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
    found = False
    engine = Tests4PyEngine(AbsoluteFitness(set(), set()), workers=32, out="rep")
    engine.evaluate(patches)
    for patch in patches:
        if almost_equal(patch.fitness, 1):
            found = True
            break
    return patches, found, duration

def main():
    approach = APPROACHES["GENPROG"]
    approach, parameters = approach
    subject = SUBJECTS["CALCULATOR"][1]
    iterations = 3
    pacthes, found, duration = evaluate(approach, subject, parameters, iterations)
    with open(f"{approach.__name__}_{subject.get_identifier()}.txt", "w") as f:
        f.write(f"{approach.__name__},{subject.get_identifier()},{found},{duration}\nFITNESS:\n{pacthes}")

if __name__ == "__main__":
    main()
