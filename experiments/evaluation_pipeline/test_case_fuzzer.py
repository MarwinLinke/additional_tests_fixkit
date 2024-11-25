from isla.fuzzer import GrammarFuzzer
from fixkit.constants import DEFAULT_WORK_DIR
from debugging_framework.benchmark.repository import BenchmarkProgram
from debugging_framework.input.oracle import OracleResult
from typing import List, Tuple, Optional
from pathlib import Path
from os import PathLike
import os
import shutil

class Tests4PyFuzzer():

    def __init__(
        self,
        benchmark_program: BenchmarkProgram,
        out: Optional[PathLike] = None,
        overwrite: Optional[bool] = False
    ):
        self.benchmark_program = benchmark_program
        self.out = out or DEFAULT_WORK_DIR
        self.overwrite = overwrite
        self.passing = []
        self.failing = []
        

    def generate_test_cases(self, num_failing: int, num_passing: int) -> Tuple[List[str]]:
        
        passing_count = 0
        failing_count = 0

        failing_inputs: List[str] = []
        passing_inputs: List[str] = []

        grammar = self.benchmark_program.get_grammar()
        oracle = self.benchmark_program.get_oracle()

        fuzzer = GrammarFuzzer(grammar)
        iteration = 0

        while iteration < 10000:

            inp = fuzzer.fuzz()
            oracle_result, _ = oracle(inp)
            iteration += 1

            if iteration % 10 == 0:
                print(f"Found {len(failing_inputs)} failing and {len(passing_inputs)} passing inputs in {iteration} iterations")


            if oracle_result == OracleResult.FAILING and inp not in failing_inputs:

                if failing_count >= num_failing:
                    continue

                failing_inputs.append(inp)
                failing_count += 1

            elif oracle_result == OracleResult.PASSING and inp not in passing_inputs:

                if passing_count >= num_passing:
                    continue

                passing_inputs.append(inp)
                passing_count += 1

            if failing_count >= num_failing and passing_count >= num_passing:
                break
        

        passing_inputs = list(set(passing_inputs))
        failing_inputs = list(set(failing_inputs))

        print(f"Failing: {len(failing_inputs)}, passing: {len(passing_inputs)} in {iteration} iterations")

        self.passing = passing_inputs
        self.failing = failing_inputs

        self.save_test_cases(self.failing, self.passing)

        return (failing_inputs, passing_inputs)

    def display_inputs(self, inputs: List[str], oracle: Tuple):
        for inp in inputs:
            print(inp.ljust(40), oracle(inp)[0])

    def save_test_cases(self, failing: List[str], passing: List[str]):
            """
            Saves each input from self.passing and self.failing as separate text files.
            Files are saved in the output directory as:
            - passing_test_X.txt
            - failing_test_X.txt
            """

            dir = self.out
            if self.overwrite:
                shutil.rmtree(dir)
            dir.mkdir(parents=True, exist_ok=True)

            for idx, test in enumerate(passing):
                passing_file_path = dir / f"passing_test_{idx}"
                with passing_file_path.open("w") as f:
                    f.write(str(test))

            for idx, test in enumerate(failing):
                failing_file_path = dir / f"failing_test_{idx}"
                with failing_file_path.open("w") as f:
                    f.write(str(test))
    
    @staticmethod
    def load_failing_test_paths(path: os.PathLike) -> List[os.PathLike]:
        """
        Retrieves failing test paths from specified directory.
        """
        filepath = Path(path)

        if filepath.exists():
            return [os.path.abspath(file) for file in filepath.glob("failing_test_*")]
        else:
            return []
        

    @staticmethod
    def load_passing_test_paths(path: os.PathLike) -> List[os.PathLike]:
        """
        Retrieves passing test paths from specified directory.
        """
        filepath = Path(path)

        if filepath.exists():
            return [os.path.abspath(file) for file in filepath.glob("passing_test_*")]
        else:
            return []



def main():
    from data import get_evaluation_data
    from tests4py.logger import LOGGER
    LOGGER.propagate = False
    setup = get_evaluation_data("GENPROG", "COOKIECUTTER", 4)
    approach, parameters, subject, benchmark_program = setup
    fuzzer = Tests4PyFuzzer(benchmark_program, out=DEFAULT_WORK_DIR / "test_cases_fuzzer")
    fuzzer.generate_test_cases(250, 250)

if __name__ == "__main__":
    main()





