from pathlib import Path
import tests4py.api as t4p
from fixkit.constants import DEFAULT_EXCLUDES
from fixkit.fitness.engine import Tests4PyEngine
from fixkit.fitness.metric import AbsoluteFitness
from fixkit.localization.t4p import Tests4PyLocalization
from fixkit.repair.pygenprog import PyGenProg
from fixkit.repair.pyae import PyAE

import time

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
    duration = time.time() - start
    found = False
    engine = Tests4PyEngine(AbsoluteFitness(set(), set()), workers=32, out="rep")
    engine.evaluate(patches)
    for patch in patches:
        print(patch.fitness)
        if almost_equal(patch.fitness, 1):
            found = True
            break
    print(found, duration)

evaluate(approach, subject, parameters)