import random
import time
import shutil
import numpy as np
from pathlib import Path
from typing import Type, Dict, Any, Tuple
import csv

import tests4py.api as t4p
from tests4py.api.test import TestResult
from tests4py.projects import Project

from fixkit.constants import DEFAULT_EXCLUDES
from fixkit.fitness.engine import Tests4PySystemTestEngine, Tests4PySystemTestSequentialEngine
from fixkit.repair.patch import get_patch
from fixkit.fitness.metric import AbsoluteFitness, GenProgFitness
from fixkit.localization.t4p import *
from fixkit.repair import GeneticRepair
from fixkit.candidate import GeneticCandidate
from fixkit.repair.pyae import PyAE
from fixkit.test_generation.test_generator import AvicennaTestGenerator, TestGenerator

from debugging_framework.benchmark.repository import BenchmarkProgram

from repair_evaluation_matrix import RepairEvaluationMatrix
from test_case_fuzzer import Tests4PyFuzzer
from data import almost_equal, get_evaluation_data, SUBJECT_PARAMS, VARIANTS

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
        use_avicenna_fault_localization: bool = False,
        use_avicenna_validation: bool = False,
        avicenna_iterations: int = 10,
        num_avicenna_tests: int = 100,
        use_cached_tests: bool = True,
        use_negated_formula: bool = True,
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
        self.use_avicenna_fault_localization = use_avicenna_fault_localization
        self.use_avicenna_validation = use_avicenna_validation
        self.avicenna_iterations = avicenna_iterations
        self.num_avicenna_tests = num_avicenna_tests
        self.use_cached_tests = use_cached_tests
        self.use_parallel_engine = use_parallel_engine
        self.use_negated_formula = use_negated_formula

        self.repair_passing_avicenna: List[str] = []
        self.repair_failing_avicenna: List[str] = []

        self.repair_passing_tests4py: List[str] = []
        self.repair_failing_tests4py: List[str] = []

        self.evaluation_passing: List[str] = []
        self.evaluation_failing: List[str] = []

        self.patches: List[GeneticCandidate] = []
        self.patch_matrices: Dict[GeneticCandidate, RepairEvaluationMatrix] = {}
        self.found: bool = False
        self.best_fitness: float = 0.0
        self.best_f1_score: float = 0.0

        self.test_case_gathering_duration = 0.0
        self.fixkit_duration = 0.0
        self.evaluation_duration = 0.0

        self.output: str = ""

    def run(self):
        start_time = time.time()

        if not self.use_avicenna_fault_localization or not self.use_avicenna_validation:
            self._gather_tests4py_tests()
        if self.use_avicenna_fault_localization or self.use_avicenna_validation:
            self._gather_avicenna_tests()
        


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

        self.repair_passing_tests4py = [
            os.path.abspath(
                os.path.join(
                    "tmp",
                    str(self.subject.get_identifier()),
                    "tests4py_systemtest_diversity",
                    f"passing_test_diversity_{i}",
                )
            ) for i in range(self.num_tests4py_tests)
        ]

        self.repair_failing_tests4py = [
            os.path.abspath(
                os.path.join(
                    "tmp",
                    str(self.subject.get_identifier()),
                    "tests4py_systemtest_diversity",
                    f"failing_test_diversity_{i}",
                )
            ) for i in range(self.num_tests4py_tests)
        ]

    def _generate_avicenna_tests(self, dir: Path, saving_path: Path):
        
        identifier: str = f"{self.subject.get_identifier()}_I{str(self.avicenna_iterations)}"
        
        default_param = {
            "max_iterations": self.avicenna_iterations,
            "saving_method": "files",
            "out": dir,
            "identifier": identifier,
            "save_automatically": False
        }

        param = self.benchmark_program.to_dict()
        param.update(default_param)

        generator = AvicennaTestGenerator(**param)

        formula = generator.load_formula(identifier)
        if not formula:
            generator.run(False)

        num_tests = self.num_avicenna_tests if self.use_negated_formula else self.num_avicenna_tests * 2

        generator.generate_more_inputs(
            max_iterations=num_tests,
            inverse_formula=False,
            only_unique_inputs=True,
            formula=formula
        )
        
        if self.use_negated_formula:
            generator.generate_more_inputs(
                max_iterations=num_tests,
                inverse_formula=True,
                only_unique_inputs=True,
                formula=formula
            )

        if saving_path.exists():
            shutil.rmtree(saving_path)

        generator.save_test_cases(saving_path)

    def _gather_avicenna_tests(self):

        report = t4p.checkout(self.subject)
        if report.raised:
            raise report.raised

        dir = os.path.abspath(Path("tmp") / "test_generator" / self.subject.get_identifier())
        saving_path = Path(dir) / "avicenna_test_cases" / f"test_cases_{self.num_avicenna_tests}"
        if not (self.use_cached_tests and saving_path.exists()):
            self._generate_avicenna_tests(dir, saving_path)

        self.repair_failing_avicenna = TestGenerator.load_failing_test_paths(saving_path)
        self.repair_passing_avicenna = TestGenerator.load_passing_test_paths(saving_path)

        print(f"Loaded {len(self.repair_failing_avicenna)} failing and {len(self.repair_passing_avicenna)} passing test cases.")

    def _repair(self):
        
        test_cases_tests4py = self.repair_passing_tests4py + self.repair_failing_tests4py
        test_cases_avicenna = self.repair_passing_avicenna + self.repair_failing_avicenna

        self.fault_localization_test_cases = test_cases_avicenna if self.use_avicenna_fault_localization else test_cases_tests4py
        self.validation_test_cases = test_cases_avicenna if self.use_avicenna_validation else test_cases_tests4py

        params = self.parameters
        if self.approach != PyAE:
            params["max_generations"] = self.fixKit_iterations

        approach = self.approach.from_source(
            src=Path("tmp", self.subject.get_identifier()),
            excludes=DEFAULT_EXCLUDES,
            localization=Tests4PySystemtestsLocalization(
                src=Path("tmp", self.subject.get_identifier()),
                events=["line"],
                predicates=["line"],
                metric="Ochiai",
                out="rep",
                tests = self.fault_localization_test_cases

            ),
            out="rep",
            is_t4p=True,
            is_system_test = True,
            system_tests = self.validation_test_cases,
            **params
        )

        self.patches: List[GeneticCandidate] = approach.repair()

    def _evaluate(self):

        print(len(self.patches))

        out = Path("evaluation_test_cases")
        fuzzer = Tests4PyFuzzer(self.benchmark_program, out, True)
        if not self.lock_evaluation_test_cases:
            fuzzer.generate_test_cases(self.num_evaluation_failing, self.num_evaluation_passing)
        self.evaluation_passing = Tests4PyFuzzer.load_passing_test_paths(out)
        self.evaluation_failing = Tests4PyFuzzer.load_failing_test_paths(out)
        evaluation_test_cases = self.evaluation_passing + self.evaluation_failing

        if self.use_parallel_engine:
            engine = Tests4PySystemTestEngine(GenProgFitness(set(self.evaluation_passing), set(self.evaluation_failing)), evaluation_test_cases, workers=32, out="rep")
        else:
            engine = Tests4PySystemTestSequentialEngine(GenProgFitness(set(self.evaluation_passing), set(self.evaluation_failing)), evaluation_test_cases, out="rep")
        engine.evaluate(self.patches)

        self.patches.sort(key = lambda patch: patch.fitness, reverse = True)

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
                
                self.best_f1_score = max(self.best_f1_score, matrix.f1_score)

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

        if self.use_avicenna_fault_localization:
            output += f"Used {len(self.repair_failing_avicenna)} failing and {len(self.repair_passing_avicenna)} passing test cases from Avicenna in the fault localization\n"
        else:
            output += f"Used {len(self.repair_failing_tests4py)} failing and {len(self.repair_passing_tests4py)} passing test cases from Tests4py in the fault localization\n"
        
        if self.use_avicenna_validation:
            output += f"Used {len(self.repair_failing_avicenna)} failing and {len(self.repair_passing_avicenna)} passing test cases from Avicenna in the validation\n"
        else:
            output += f"Used {len(self.repair_failing_tests4py)} failing and {len(self.repair_passing_tests4py)} passing test cases from Tests4py in the validation\n"

        output += f"In total {len(self.fault_localization_test_cases)} for fault localization and {len(self.validation_test_cases)} for validation.\n"
        output += f"The gathering of test cases took {"{:.4f}".format(self.test_case_gathering_duration)} seconds.\n"
        output += f"The repair ran for {"{:.4f}".format(self.fixkit_duration)} seconds.\n"
        output += f"The evaluation took {"{:.4f}".format(self.evaluation_duration)} seconds.\n"
        output += f"Was a valid patch found: {self.found}\n"
        output += f"BEST FITNESS: {self.best_fitness}\n"
        output += f"BEST F1 SCORE: {self.best_f1_score}\n"
        output += f"Found a total of {len(self.patches)} patches.\n\n"
        output += "".ljust(100, "%")
        output += f"\n\nPATCHES (SORTED):\n"

        for patch in self.patches:
            output += str(patch)
            matrix = str(self.patch_matrices[patch] or "\nNo tests4py report was found, matrix could not be calculated.")
            output += matrix
            fix = get_patch(patch)
            if fix == "":
                output += "\nPatch could not be printed.\n\n"
            else:
                output += f"\n{fix}\n"
            output += f"{"".ljust(100, "_")}\n\n"

        self.output = output

    def write_to_csv(self, file: str):
        with open(file, "a", newline="") as file:
            writer = csv.writer(file)
            variant = "C" if self.use_avicenna_fault_localization and self.use_avicenna_validation else "F" if self.use_avicenna_fault_localization else "V" if self.use_avicenna_validation else "B"
            best_patch = self.patches[0]
            matrix = self.patch_matrices[best_patch]
            precision = matrix.precision if matrix else None
            recall = matrix.recall if matrix else None
            f1_score = matrix.f1_score if matrix else None
            accuracy = matrix.accuracy if matrix else None
            engine = "parallel" if self.use_parallel_engine else "sequential"
            data = [self.approach.__name__, self.subject.project_name, self.subject.bug_id, 
                    self.fixKit_iterations, engine, variant, 
                    len(self.repair_failing_avicenna), len(self.repair_passing_avicenna),
                    len(self.repair_failing_tests4py), len(self.repair_passing_tests4py),
                    len(self.fault_localization_test_cases), len(self.validation_test_cases),
                    self.test_case_gathering_duration, self.fixkit_duration, self.evaluation_duration,
                    self.best_fitness, precision, recall, f1_score, accuracy, len(self.patches)]
            writer.writerow(data)

    @staticmethod
    def write_csv_header(file: str):
        with open(file, "w", newline="") as file:
            writer = csv.writer(file)
            header =["approach", "subject", "bug_id", "iterations", "engine", "variant", "failing_avicenna", "passing_avicenna", 
                "failing_tests4py", "passing_tests4py", "fault_localization_tests", "validation_tests",
                "gathering_duration", "repair_duration", "evaluation_duration", "best_fitness", 
                "precision", "recall", "f1_score", "accuracy", "patches"]
            writer.writerow(header)
            


def main():
    if Path("rep").exists():
        shutil.rmtree("rep")

    setup = get_evaluation_data("GENPROG", "MARKUP", 1)
    approach, parameters, subject, benchmark_program = setup
    print(approach, parameters, subject, benchmark_program)
    iterations = 3
    num_tests4py_tests = 1
    use_avicenna_fault_localization = True
    use_avicenna_validation = True
    use_cached_tests = True
    use_negated_formula = False
    avicenna_iterations = 10
    num_avicenna_tests = 50
    use_parallel_engine = False

    pipeline = EvaluationPipeline(
        approach=approach,
        parameters=parameters,
        subject=subject,
        benchmark_program=benchmark_program,
        fixkit_iterations=iterations,
        num_tests4py_tests=num_tests4py_tests,
        use_avicenna_fault_localization=use_avicenna_fault_localization,
        use_avicenna_validation=use_avicenna_validation,
        avicenna_iterations=avicenna_iterations,
        num_avicenna_tests=num_avicenna_tests,
        use_cached_tests=use_cached_tests,
        use_negated_formula=use_negated_formula,
        use_parallel_engine=use_parallel_engine
    )

    pipeline.run()
    #pipeline.write_csv_header("tmp/data.csv")
    pipeline.write_to_csv("tmp/data.csv")

    variant = "C" if use_avicenna_fault_localization and use_avicenna_validation else "F" if use_avicenna_fault_localization else "V" if use_avicenna_validation else "B"
    tests_identifier_avicenna = f"A{num_avicenna_tests}" if variant != "B" else ""
    tests_identifier_tests4py = f"T{num_tests4py_tests}" if variant != "C" else ""
    engine_identifier = "P" if use_parallel_engine else "S"
    file_name = f"{approach.__name__}_{subject.project_name}{subject.bug_id}_I{iterations}-{variant}-{tests_identifier_avicenna}{tests_identifier_tests4py}-{engine_identifier}"

    out = Path("out", f"{file_name}.txt")
    with open(out, "w") as f:
        f.write(pipeline.output)


if __name__ == "__main__":
    main()