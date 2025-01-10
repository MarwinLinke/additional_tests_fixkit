from typing import *

import tests4py.api as t4p
from tests4py.projects import Project
from debugging_benchmark.calculator.calculator import CalculatorBenchmarkRepository
from debugging_benchmark.middle.middle import MiddleBenchmarkRepository
from debugging_benchmark.markup.markup import MarkupBenchmarkRepository
from debugging_benchmark.expression.expression import ExpressionBenchmarkRepository
from debugging_benchmark.tests4py_benchmark.repository import PysnooperBenchmarkRepository
from debugging_benchmark.tests4py_benchmark.repository import CookiecutterBenchmarkRepository
from debugging_benchmark.tests4py_benchmark.repository import Tests4PyBenchmarkRepository
from debugging_framework.benchmark.repository import BenchmarkProgram
from fixkit.repair.pygenprog import PyGenProg
from fixkit.repair import GeneticRepair
from fixkit.localization.modifier import SigmoidModifier, TopRankModifier, TopEqualRankModifier, DefaultModifier


APPROACHES: Dict[str, Tuple[Type[GeneticRepair], Dict[str, float]]] = {
    "GENPROG": (
        PyGenProg,
        {
            "modifier": TopEqualRankModifier(),
            "population_size": 40,
            "w_mut": 0.2,
            "workers": 32,
        },
    )
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
        "ITERATIONS": [10],
        "NUM_BASELINE_TESTS": [(1, 1), (1, 10)],
        "NUM_ADDITIONAL_TESTS": [(5, 5), (10, 10), (30, 30), (50, 50)],
    },

    "MIDDLE_2": {
        "SUBJECT": "MIDDLE",
        "BUG_ID": 2,
        "ITERATIONS": [10],
        "NUM_BASELINE_TESTS": [(1, 1), (1, 10)],
        "NUM_ADDITIONAL_TESTS": [(5, 5), (10, 10), (30, 30), (50, 50)],
    },

    "MARKUP_1": {
        "SUBJECT": "MARKUP",
        "BUG_ID": 1,
        "ITERATIONS": [10],
        "NUM_BASELINE_TESTS": [(1, 1), (1, 10)],
        "NUM_ADDITIONAL_TESTS": [(5, 5), (10, 10), (30, 30), (50, 50)],
    },

    "MARKUP_2": {
        "SUBJECT": "MARKUP",
        "BUG_ID": 2,
        "ITERATIONS": [10],
        "NUM_BASELINE_TESTS": [(1, 1), (1, 10)],
        "NUM_ADDITIONAL_TESTS": [(5, 5), (10, 10), (30, 30), (50, 50)],
    },

    "EXPRESSION_1": {
        "SUBJECT": "EXPRESSION",
        "BUG_ID": 1,
        "ITERATIONS": [10],
        "NUM_BASELINE_TESTS": [(1, 1), (1, 10)],
        "NUM_ADDITIONAL_TESTS": [(5, 5), (10, 10), (30, 30), (50, 50)],
    }, 

    "CALCULATOR_1": {
        "SUBJECT": "CALCULATOR",
        "BUG_ID": 1,
        "ITERATIONS": [10],
        "NUM_BASELINE_TESTS": [(1, 1), (1, 10)],
        "NUM_ADDITIONAL_TESTS": [(5, 5), (10, 10), (30, 30), (50, 50)],
    }, 

    "PYSNOOPER_2": {
        "SUBJECT": "PYSNOOPER",
        "BUG_ID": 2,
        "ITERATIONS": [10],
        "NUM_BASELINE_TESTS": [(1, 1), (1, 10)],
        "NUM_ADDITIONAL_TESTS": [(5, 5), (10, 10), (30, 30), (50, 50)],
    },

    
    "PYSNOOPER_3": {
        "SUBJECT": "PYSNOOPER",
        "BUG_ID": 3,
        "ITERATIONS": [10],
        "NUM_BASELINE_TESTS": [(1, 1), (1, 10)],
        "NUM_ADDITIONAL_TESTS": [(5, 5), (10, 10), (30, 30), (50, 50)],
    },

    "DEFAULT": {
        "ITERATIONS": [10],
        "NUM_BASELINE_TESTS": [(1, 1), (1, 10)],
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
    ) -> Tuple[Type[GeneticRepair], Dict[str, float], Project]:

    approach_data = APPROACHES[approach_identifier]
    subject_data = SUBJECTS[subject_identifier][bug_id]
    approach, parameters = approach_data
    subject, _ = subject_data

    return approach, parameters, subject


def get_benchmark_program(subject: str, bug_id: int) -> BenchmarkProgram:
    subject_data = SUBJECTS[subject][bug_id]
    _, benchmark_callable = subject_data
    benchmark_program = benchmark_callable()
    return benchmark_program
    

def get_subject_params(subject: str) -> Dict:
    if subject in SUBJECT_PARAMS:
        return SUBJECT_PARAMS[subject]
    else:

        subject_id = subject.split("_")
        subject_name = subject_id[0]
        bug_id = int(subject_id[1])

        if subject_name not in SUBJECTS:
            raise ValueError(f"Subject of name {subject_name} was not found in valid subjects.")
        
        if bug_id not in SUBJECTS[subject_name]:
            raise ValueError(f"The bug with the id {bug_id} was not found for {subject_name}.")

        identifier = {"SUBJECT": subject_name, "BUG_ID": bug_id}
        params = SUBJECT_PARAMS["DEFAULT"]
        params.update(identifier)
        return params

def almost_equal(value, target, delta=0.0001):
    return abs(value - target) < delta