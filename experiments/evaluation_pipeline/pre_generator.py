from fixkit.test_generation.fuzzer_test_generator import GrammarFuzzerTestGenerator
from debugging_framework.benchmark.repository import BenchmarkProgram
from pathlib import Path
import shutil
from os import PathLike

class PreGenerator():

    def __init__(
        self,
        subject_name: str,
        benchmark_program: BenchmarkProgram,
        num_failing: int,
        num_passing: int,
        path: PathLike,
        seed: int
    ):
        self.subject_name = subject_name
        self.benchmark_program: BenchmarkProgram = benchmark_program
        self.num_failing = num_failing
        self.num_passing = num_passing
        self.seed = seed
        self.path = path

    def run(self) -> str:

        dir = self.path
        saving_path = Path(dir) / f"{self.subject_name}_{self.num_failing}-{self.num_passing}_{self.seed}"

        generator = GrammarFuzzerTestGenerator(
            oracle = self.benchmark_program.get_oracle(),
            grammar = self.benchmark_program.get_grammar(),
            num_failing = self.num_failing,
            num_passing = self.num_passing,
            out = dir,
            saving_method = "files",
            save_automatically = False,
            seed = self.seed 
        )

        generator.run()

        if saving_path.exists():
            shutil.rmtree(saving_path)

        generator.save_test_cases(saving_path)
        
        return saving_path