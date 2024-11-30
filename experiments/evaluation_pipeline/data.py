from debugging_benchmark.calculator.calculator import CalculatorBenchmarkRepository
from debugging_benchmark.middle.middle import MiddleBenchmarkRepository
from debugging_benchmark.markup.markup import MarkupBenchmarkRepository
from debugging_benchmark.expression.expression import ExpressionBenchmarkRepository
from debugging_benchmark.tests4py_benchmark.repository import PysnooperBenchmarkRepository
from debugging_benchmark.tests4py_benchmark.repository import CookiecutterBenchmarkRepository
from debugging_framework.benchmark.repository import BenchmarkProgram

from fixkit.repair.pygenprog import PyGenProg
from fixkit.repair.pykali import PyKali
from fixkit.repair.pymutrepair import PyMutRepair
from fixkit.repair.pycardumen import PyCardumen
from fixkit.repair.pyae import PyAE
from fixkit.repair import GeneticRepair

import tests4py.api as t4p
from tests4py.projects import Project

from typing import *

APPROACHES: Dict[str, Tuple[Type[GeneticRepair], Dict[str, float]]] = {
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

# SUBJECTS is using lambda calls to avoid building all benchmark repositories during initialization
SUBJECTS: Dict[str, Dict[int, Tuple[Project, Callable[[], BenchmarkProgram]]]] = {
    "MIDDLE": {
        1: (t4p.middle_1, lambda: MiddleBenchmarkRepository().build()[0]),
        2: (t4p.middle_2, lambda: MiddleBenchmarkRepository().build()[0])
    },

    "MARKUP": {
        1: (t4p.markup_1, lambda: MarkupBenchmarkRepository().build()[0]),
        2: (t4p.markup_2, lambda: MarkupBenchmarkRepository().build()[0])
    },

    "EXPRESSION": {
        1: (t4p.expression_1, lambda: ExpressionBenchmarkRepository().build()[0]),
    }, 

    "CALCULATOR": {
        1: (t4p.calculator_1, lambda: CalculatorBenchmarkRepository().build()[0]),
    }, 

    "PYSNOOPER": {
        1: (t4p.pysnooper_1, lambda: None),
        2: (t4p.pysnooper_2, lambda: PysnooperBenchmarkRepository().build()[0]),
        3: (t4p.pysnooper_3, lambda: PysnooperBenchmarkRepository().build()[1]),
    },

    "COOKIECUTTER": {
        1: (t4p.cookiecutter_1, lambda: None),
        2: (t4p.cookiecutter_2, lambda: CookiecutterBenchmarkRepository().build()[0]),
        3: (t4p.cookiecutter_3, lambda: CookiecutterBenchmarkRepository().build()[1]),
        4: (t4p.cookiecutter_4, lambda: CookiecutterBenchmarkRepository().build()[2]),
    },
}


SUBJECT_PARAMS = {
    "MIDDLE_1": {
        "SUBJECT": "MIDDLE",
        "BUG_ID": 1,
        "ITERATIONS": [1, 10],
        "NUM_BASELINE_TESTS": [(1, 10)],
        "NUM_ADDITIONAL_TESTS": [(5, 5), (10, 10), (30, 30), (50, 50)],
    },

    "MIDDLE_2": {
        "SUBJECT": "MIDDLE",
        "BUG_ID": 2,
        "ITERATIONS": [1, 10],
        "NUM_BASELINE_TESTS": [(1, 10)],
        "NUM_ADDITIONAL_TESTS": [(5, 5), (10, 10), (30, 30), (50, 50)],
    },

    "MARKUP_1": {
        "SUBJECT": "MARKUP",
        "BUG_ID": 1,
        "ITERATIONS": [1, 10],
        "NUM_BASELINE_TESTS": [(1, 10)],
        "NUM_ADDITIONAL_TESTS": [(5, 5), (10, 10), (30, 30), (50, 50)],
    },

    "MARKUP_2": {
        "SUBJECT": "MARKUP",
        "BUG_ID": 2,
        "ITERATIONS": [1, 10],
        "NUM_BASELINE_TESTS": [(1, 10)],
        "NUM_ADDITIONAL_TESTS": [(5, 5), (10, 10), (30, 30), (50, 50)],
    },

    "EXPRESSION_1": {
        "SUBJECT": "EXPRESSION",
        "BUG_ID": 1,
        "ITERATIONS": [1, 10],
        "NUM_BASELINE_TESTS": [(1, 10)],
        "NUM_ADDITIONAL_TESTS": [(5, 5), (10, 10), (30, 30), (50, 50)],
    }, 

    "CALCULATOR_1": {
        "SUBJECT": "CALCULATOR",
        "BUG_ID": 1,
        "ITERATIONS": [1, 10],
        "NUM_BASELINE_TESTS": [(1, 10)],
        "NUM_ADDITIONAL_TESTS": [(5, 5), (10, 10), (30, 30), (50, 50)],
    }, 

    "PYSNOOPER_2": {
        "SUBJECT": "PYSNOOPER",
        "BUG_ID": 2,
        "ITERATIONS": [1, 10],
        "NUM_BASELINE_TESTS": [(1, 10)],
        "NUM_ADDITIONAL_TESTS": [(5, 5), (10, 10), (30, 30), (50, 50)],
    },

    
    "PYSNOOPER_3": {
        "SUBJECT": "PYSNOOPER",
        "BUG_ID": 3,
        "ITERATIONS": [1, 10],
        "NUM_BASELINE_TESTS": [(1, 10)],
        "NUM_ADDITIONAL_TESTS": [(5, 5), (10, 10), (30, 30), (50, 50)],
    },

    "COOKIECUTTER_3": {
        "SUBJECT": "COOKIECUTTER",
        "BUG_ID": 3,
        "ITERATIONS": [1, 10],
        "NUM_BASELINE_TESTS": [(1, 10)],
        "NUM_ADDITIONAL_TESTS": [(5, 5), (10, 10), (30, 30), (50, 50)],
    },
}


VARIANTS = {
    "BASELINE": "NUM_BASELINE_TESTS", 
    "FAULT_LOCALIZATION": "NUM_ADDITIONAL_TESTS", 
    "VALIDATION": "NUM_ADDITIONAL_TESTS", 
    "COMPLETE" : "NUM_ADDITIONAL_TESTS"
}

def get_evaluation_data(
        approach_identifier: str, 
        subject_identifier: str, 
        bug_id: int
    ) -> Tuple[Type[GeneticRepair], Dict[str, float], Project, BenchmarkProgram]:

    approach_data = APPROACHES[approach_identifier]
    subject_data = SUBJECTS[subject_identifier][bug_id]
    approach, parameters = approach_data
    subject, benchmark_callable = subject_data
    benchmark_program = benchmark_callable()

    return approach, parameters, subject, benchmark_program


def almost_equal(value, target, delta=0.0001):
    return abs(value - target) < delta