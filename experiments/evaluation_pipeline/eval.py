import random
import time
import shutil
import numpy as np
from pathlib import Path
from typing import Type, Dict, Any, Tuple

import tests4py.api as t4p
from tests4py.api.test import TestResult
from tests4py.projects import Project

from fixkit.constants import DEFAULT_EXCLUDES
from fixkit.fitness.engine import Tests4PySystemTestEngine, Tests4PySystemTestSequentialEngine
from fixkit.repair.patch import get_patch
from fixkit.fitness.metric import AbsoluteFitness
from fixkit.localization.t4p import *
from fixkit.repair import GeneticRepair
from fixkit.candidate import GeneticCandidate

from debugging_framework.benchmark.repository import BenchmarkProgram

from repair_evaluation_matrix import RepairEvaluationMatrix
from test_case_fuzzer import Tests4PyFuzzer
from data import almost_equal, get_evaluation_data

random.seed(0)
np.random.seed(0)

class EvaluationPipeline():

    def __init__(
        self,
        approach: Type[GeneticRepair], 
        parameters: Dict[str, Any], 
        subject: Project,
        benchmark_program: BenchmarkProgram,
        fixkit_iterations: int,
        num_tests4py_tests: int,
        num_evaluation_failing: int = 50,
        num_evaluation_passing: int = 50,
        lock_evaluation_test_cases: bool = False,
        use_avicenna: bool = False,
        use_parallel_engine: bool = False
    ):
        self.approach: Type[GeneticRepair] = approach
        self.parameters: Dict[str, Any] = parameters
        self.subject: Project = subject
        self.benchmark_program: BenchmarkProgram = benchmark_program
        self.fixKit_iterations = fixkit_iterations
        self.num_tests4py_tests = num_tests4py_tests
        self.num_evaluation_passing = num_evaluation_passing
        self.num_evaluation_failing = num_evaluation_failing
        self.lock_evaluation_test_cases = lock_evaluation_test_cases
        self.use_avicenna = use_avicenna
        self.use_parallel_engine = use_parallel_engine

        self.repair_passing: List[str] = []
        self.repair_failing: List[str] = []

        self.evaluation_passing: List[str] = []
        self.evaluation_failing: List[str] = []

        self.patches: List[GeneticCandidate] = []
        self.patch_matrices: Dict[GeneticCandidate, RepairEvaluationMatrix] = {}
        self.found: bool = False
        self.best_fitness: float = 0.0

        self.test_case_gathering_duration = 0.0
        self.fixkit_duration = 0.0
        self.evaluation_duration = 0.0

        self.output: str = ""

    def run(self):
        start_time = time.time()

        if self.use_avicenna:
            self._gather_avicenna_tests()
        else:
            self._gather_tests4py_tests()

        self.test_case_gathering_duration = time.time() - start_time

        self._repair()

        self.fixkit_duration = time.time() - start_time - self.test_case_gathering_duration

        self._evaluate()

        self.evaluation_duration = time.time() - start_time - self.test_case_gathering_duration - self.fixkit_duration
        
        self._format_output()

    def _gather_tests4py_tests(self):

        print(self.subject)
        print(type(self.subject))

        report = t4p.checkout(self.subject)
        if report.raised:
            raise report.raised

        self.repair_passing = [
            os.path.abspath(
                os.path.join(
                    "tmp",
                    str(self.subject.get_identifier()),
                    "tests4py_systemtest_diversity",
                    f"passing_test_diversity_{i}",
                )
            ) for i in range(self.num_tests4py_tests)
        ]

        self.repair_failing = [
            os.path.abspath(
                os.path.join(
                    "tmp",
                    str(self.subject.get_identifier()),
                    "tests4py_systemtest_diversity",
                    f"failing_test_diversity_{i}",
                )
            ) for i in range(self.num_tests4py_tests)
        ]

    def _gather_avicenna_tests(self):
        raise NotImplementedError("Avicenna not implemented yet.")

    def _repair(self):
        
        test_cases = self.repair_passing + self.repair_failing

        approach = self.approach.from_source(
            src=Path("tmp", self.subject.get_identifier()),
            excludes=DEFAULT_EXCLUDES,
            localization=Tests4PySystemtestsLocalization(
                src=Path("tmp", self.subject.get_identifier()),
                events=["line"],
                predicates=["line"],
                metric="Ochiai",
                out="rep",
                tests = test_cases

            ),
            out="rep",
            is_t4p=True,
            max_generations=self.fixKit_iterations,
            is_system_test = True,
            system_tests = test_cases,
            **self.parameters,
        )

        self.patches: List[GeneticCandidate] = approach.repair()

    def _evaluate(self):

        out = Path("evaluation_test_cases")
        fuzzer = Tests4PyFuzzer(self.benchmark_program, out, True)
        if not self.lock_evaluation_test_cases:
            fuzzer.generate_test_cases(self.num_evaluation_failing, self.num_evaluation_passing)
        self.evaluation_passing = Tests4PyFuzzer.load_passing_test_paths(out)
        self.evaluation_failing = Tests4PyFuzzer.load_failing_test_paths(out)
        evaluation_test_cases = self.evaluation_passing + self.evaluation_failing

        if self.use_parallel_engine:
            engine = Tests4PySystemTestEngine(AbsoluteFitness(set(), set()), evaluation_test_cases, workers=32, out="rep")
        else:
            engine = Tests4PySystemTestSequentialEngine(AbsoluteFitness(set(), set()), evaluation_test_cases, out="rep")
        engine.evaluate(self.patches)
        print(len(self.patches))

        for patch in self.patches:

            if patch.tests4py_report:
                passing: List[str] = []
                failing: List[str] = []
                results: Dict[str, Tuple[TestResult, str]] = patch.tests4py_report.results

                for test, (result, _) in results.items():
                    if result == TestResult.PASSING:
                        passing.append(test)
                    elif result == TestResult.FAILING:
                        failing.append(test)

                matrix = RepairEvaluationMatrix(self.evaluation_passing, self.evaluation_failing, passing, failing)
                self.patch_matrices[patch] = matrix

                print(str(patch))
                print(str(matrix))
            else:
                self.patch_matrices[patch] = None

            self.best_fitness = max(self.best_fitness, patch.fitness)
            if almost_equal(patch.fitness, 1):
                self.found = True

    def _format_output(self):
        output: str = f"APPROACH: {self.approach.__name__}\n"
        output += f"SUBJECT: {self.subject.get_identifier()}\n"
        output += f"The repair ran for {"{:.4f}".format(self.test_case_gathering_duration)} seconds.\n"
        output += f"The repair ran for {"{:.4f}".format(self.fixkit_duration)} seconds.\n"
        output += f"The evaluation took {"{:.4f}".format(self.evaluation_duration)} seconds.\n"
        output += f"Was a valid patch found: {self.found}\n"
        output += f"BEST FITNESS: {self.best_fitness}\n"
        output += f"Found a total of {len(self.patches)} patches.\n"
        output += f"\nPATCHES:\n"

        for patch in self.patches:
            output += str(patch)
            matrix = str(self.patch_matrices[patch]) or "No tests4py was found."
            output += matrix
            fix = get_patch(patch)
            if fix == "":
                output += "\nPatch couldn't be printed."
            else:
                output += f"\n{fix}\n"

        self.output = output


def main():
    if Path("rep").exists():
        shutil.rmtree("rep")

    setup = get_evaluation_data("GENPROG", "MIDDLE", 2)
    approach, parameters, subject, benchmark_program = setup
    iterations = 3
    num_tests4py_tests = 3
    use_avicenna = False
    use_parallel_engine = True

    pipeline = EvaluationPipeline(
        approach=approach,
        parameters=parameters,
        subject=subject,
        benchmark_program=benchmark_program,
        fixkit_iterations=iterations,
        num_tests4py_tests=num_tests4py_tests,
        use_avicenna=use_avicenna,
        use_parallel_engine=use_parallel_engine
    )

    pipeline.run()
    
    test_cases_identifier = "A" if use_avicenna else f"F{num_tests4py_tests}"
    engine_identifier = "P" if use_parallel_engine else "S"
    file_name = f"{approach.__name__}_{subject.project_name}{subject.bug_id}_I{iterations}-{test_cases_identifier}-{engine_identifier}"

    out = Path("out", f"{file_name}.txt")
    with open(out, "w") as f:
        f.write(pipeline.output)

if __name__ == "__main__":
    main()
