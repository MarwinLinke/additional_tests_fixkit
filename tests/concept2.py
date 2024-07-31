from pathlib import Path
import tests4py.api as t4p
from fixkit.repair.pygenprog import PyGenProg
from fixkit.constants import DEFAULT_EXCLUDES
from fixkit.fitness.engine import Tests4PyEngine
from fixkit.fitness.metric import AbsoluteFitness
from fixkit.localization.t4p import Tests4PyLocalization
from fixkit.repair.pyae import PyAE


from isla.solver import ISLaSolver
import string

from avicenna.avicenna import OracleResult
from avicenna.avicenna import Avicenna
from typing import List, Tuple
from isla.language import Formula
from isla.language import ISLaUnparser

import time
import math

subject = t4p.calculator_1
approach = PyAE
parameters = {"k": 1}


def almost_equal(value, target, delta=0.0001):
    return abs(value - target) < delta


def evaluate(approach, subject, parameters):
    report = t4p.checkout(subject)
    if report.raised:
        raise report.raised
    start = time.time()
    print("CREATED REPORT, START REPAIRING")
    approach = approach.from_source(
        src=Path("tmp", subject.get_identifier()),
        excludes=DEFAULT_EXCLUDES,
        localization=Tests4PyLocalization(
            src=Path("tmp", subject.get_identifier()),
            events=["line"],
            predicates=["line"],
            metric="Ochiai",
            out="rep",
        ),
        out="rep",
        is_t4p=True,
        **parameters,
    )
    patches = approach.repair()
    print("REPAIR COMPLETE, START T4P ENGINE")
    duration = time.time() - start
    found = False
    engine = Tests4PyEngine(AbsoluteFitness(set(), set()), workers=32, out="rep")
    print("T4P ENGINE COMPLETE INIT, START EVALUATING")
    engine.evaluate(patches)
    print("EVALUATION COMPLETE, GOING THROUGH PATCHES")
    for patch in patches:
        print(patch.fitness)
        if almost_equal(patch.fitness, 1):
            found = True
            break
    print(found, duration)

evaluate(approach, subject, parameters)