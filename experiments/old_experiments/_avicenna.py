from avicenna import Avicenna
from avicenna.runner.report import SingleFailureReport
from debugging_benchmark.calculator.calculator import CalculatorBenchmarkRepository


if __name__ == "__main__":
    default_param = {
        "max_iterations": 10,
    }

    calculator_subject = CalculatorBenchmarkRepository().build()
    param = calculator_subject[0].to_dict()
    param.update(default_param)

    avicenna = Avicenna(**param, enable_logging=True)
    diagnoses = avicenna.explain()
    print(diagnoses)