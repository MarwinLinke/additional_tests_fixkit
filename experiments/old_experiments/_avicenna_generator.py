from fixkit.test_generation.test_generator import AvicennaTestGenerator, TestGenerator
from debugging_benchmark.calculator.calculator import CalculatorBenchmarkRepository
from debugging_benchmark.middle.middle import MiddleBenchmarkRepository
from debugging_benchmark.tests4py_benchmark.repository import PysnooperBenchmarkRepository

default_param = {
    "max_iterations": 5,
    "saving_method": "files",
    "overwrite": True
}

subject = CalculatorBenchmarkRepository().build()
param = subject[1].to_dict()
param.update(default_param)

generator = AvicennaTestGenerator(**param)
generator.run()
path = generator.out
generator.generate_more_inputs(100)

print()
print(generator.diagnosis)
print()
print(TestGenerator.load_failing_test_paths(path))
