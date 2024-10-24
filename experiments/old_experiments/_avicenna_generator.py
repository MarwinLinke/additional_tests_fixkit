from fixkit.test_generation.test_generator import AvicennaTestGenerator, TestGenerator
from debugging_benchmark.calculator.calculator import CalculatorBenchmarkRepository

default_param = {
    "max_iterations": 5,
    "saving_method": "files"
}


subject = CalculatorBenchmarkRepository().build()
param = subject[0].to_dict()
param.update(default_param)


generator = AvicennaTestGenerator(**param)
generator.run()
path = generator.out
print(path)
print(TestGenerator.load_failing_test_paths(path))
