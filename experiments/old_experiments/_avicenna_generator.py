from fixkit.test_generation.test_generator import AvicennaTestGenerator, TestGenerator
from debugging_benchmark.calculator.calculator import CalculatorBenchmarkRepository
from debugging_benchmark.middle.middle import MiddleBenchmarkRepository
from debugging_benchmark.tests4py_benchmark.repository import PysnooperBenchmarkRepository
from isla.language import ISLaUnparser

default_param = {
    "max_iterations": 5,
    "saving_method": "files",
    "overwrite": True
}

subject = PysnooperBenchmarkRepository().build()
param = subject[0].to_dict()
param.update(default_param)

print(param)

generator = AvicennaTestGenerator(**param)
generator.run()
path = generator.out
generator.generate_more_inputs(100, False)

print()
print(ISLaUnparser(generator.diagnoses.pop(0).formula).unparse())
print()
print(TestGenerator.load_failing_test_paths(path))
