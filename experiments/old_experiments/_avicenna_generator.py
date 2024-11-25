from fixkit.test_generation.test_generator import AvicennaTestGenerator, TestGenerator
from debugging_benchmark.calculator.calculator import CalculatorBenchmarkRepository
from debugging_benchmark.middle.middle import MiddleBenchmarkRepository
from debugging_benchmark.markup.markup import MarkupBenchmarkRepository
from debugging_benchmark.expression.expression import ExpressionBenchmarkRepository
from debugging_benchmark.tests4py_benchmark.repository import PysnooperBenchmarkRepository
from debugging_benchmark.tests4py_benchmark.repository import CookiecutterBenchmarkRepository
from isla.language import ISLaUnparser
from pathlib import Path
import logging
from tests4py.logger import LOGGER

default_param = {
    "max_iterations": 10,
    "saving_method": "files",
    "identifier": "cookiecutter_10"
}

subject = CookiecutterBenchmarkRepository().build()
param = subject[1].to_dict()
param.update(default_param)

print(param)

LOGGER.propagate = False

generator = AvicennaTestGenerator(**param)
#generator.run(False)
path = generator.out / "test_cases"
formula = generator.load_formula("cookiecutter_10")
print(formula)
generator.generate_more_inputs(250, False, formula, True, False)
generator.generate_more_inputs(250, True, formula, True, False)

print(TestGenerator.load_failing_test_paths(path))
