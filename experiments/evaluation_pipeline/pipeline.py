import random
import time
import shutil
import csv
import os
import numpy as np

from pathlib import Path
from typing import Type, Dict, Any, Tuple, List
from enum import Enum

import tests4py.api as t4p
from tests4py.api.test import TestResult
from tests4py.projects import Project

from fixkit.constants import DEFAULT_EXCLUDES
from fixkit.fitness.engine import Tests4PySystemTestEngine, Tests4PySystemTestSequentialEngine
from fixkit.repair.patch import get_patch
from fixkit.fitness.metric import GenProgFitness
from fixkit.localization.t4p import Tests4PySystemtestsLocalization
from fixkit.repair import GeneticRepair
from fixkit.candidate import GeneticCandidate
from fixkit.repair.pyae import PyAE
from fixkit.test_generation.test_generator import TestGenerator
from fixkit.test_generation.fuzzer_test_generator import GrammarFuzzerTestGenerator
from fixkit.test_generation.avicenna_test_generator import AvicennaTestGenerator
from fixkit.logger import LOGGER

from debugging_framework.benchmark.repository import BenchmarkProgram

from repair_evaluation_matrix import RepairEvaluationMatrix
from data import almost_equal

class TestGenerationType(Enum):
    AVICENNA = 0
    GRAMMAR_FUZZER = 1

class EvaluationPipeline():

    def __init__(
        self,
        approach: Type[GeneticRepair], 
        parameters: Dict[str, Any], 
        subject: Project,
        benchmark_program: BenchmarkProgram,
        repair_iterations: int,
        seed: int,
        test_generation_type: TestGenerationType = TestGenerationType.GRAMMAR_FUZZER,
        num_baseline_failing: int = 1,
        num_baseline_passing: int = 10,
        num_additional_failing: int = 50,
        num_additional_passing: int = 50,
        num_evaluation_failing: int = 50,
        num_evaluation_passing: int = 50,
        lock_evaluation_tests: bool = False,
        enhance_fault_localization: bool = False,
        enhance_validation: bool = False,
        use_cached_tests: bool = True,
        use_parallel_engine: bool = False
    ):
        """
        :param Type[GeneticRepair] approach: The used repair approach.
        :param Project subject: The tests4py subject.
        :param int repair_iterations: The number of iterations the repair will run for.
        :param int seed: The seed used for random and numpy.random.
        :param TestGenerationType test_generation_type: The type of additional test case generation. Use AVICENNA or GRAMMAR_FUZZER.
        :param int num_baseline_failing: The number of failing test cases used for the baseline.
        :param int num_baseline_passing: The number of passing test cases used for the baseline.
        :param int num_additional_failing: The number of additional failing test cases used.
        :param int num_additional_passing: The number of additional passing test cases used.
        :param int num_evaluation_failing: The number of failing test cases used for evaluating the quality of patches.
        :param int num_evaluation_passing: The number of passing test cases used for evaluating the quality of patches.
        :param bool lock_evaluation_tests: If this is true, no evaluation test cases are generated. Useful for evaluating multiple runs on the same test cases.
        :param bool enhance_fault_localization: If this is true, additional test cases will be used in the fault localization.
        :param bool enhance_validation: If this is true, additional test cases will be used in the validation.
        :param bool use_cached_tests: If this is true, no additional test cases are generated if they already exist.
        :param bool use_parallel_engine: Uses either the parallel or sequential engine for evaluating the patches.
        """
        self.approach: Type[GeneticRepair] = approach
        self.parameters: Dict[str, Any] = parameters
        self.subject: Project = subject
        self.benchmark_program: BenchmarkProgram = benchmark_program
        self.fixKit_iterations = repair_iterations
        self.test_generation_type = test_generation_type

        self.num_baseline_failing = num_baseline_failing
        self.num_baseline_passing = num_baseline_passing
        self.num_additional_failing = num_additional_failing
        self.num_additional_passing = num_additional_passing
        self.num_evaluation_failing = num_evaluation_failing
        self.num_evaluation_passing = num_evaluation_passing

        self.lock_evaluation_test_cases = lock_evaluation_tests
        self.enhance_fault_localization = enhance_fault_localization
        self.enhance_validation = enhance_validation
        self.use_cached_tests = use_cached_tests
        self.use_parallel_engine = use_parallel_engine

        self.repair_additional_failing: List[str] = []
        self.repair_additional_passing: List[str] = []

        self.repair_baseline_failing: List[str] = []
        self.repair_baseline_passing: List[str] = []

        self.evaluation_failing: List[str] = []
        self.evaluation_passing: List[str] = []

        self.seed = seed
        self._set_seed()
        self.patches: List[GeneticCandidate] = []
        self.patch_matrices: Dict[GeneticCandidate, RepairEvaluationMatrix] = {}
        self.found: bool = False
        self.best_fitness: float = 0.0
        self.best_f1_score: float = 0.0

        self.start_time = 0.0
        self.collection_duration = 0.0
        self.repair_duration = 0.0
        self.evaluation_duration = 0.0

        self.output: str = ""

    def _set_seed(self):      
        random.seed(self.seed)
        np.random.seed(self.seed)

    def run(self):
        """
        Runs an evaluation pipeline with custom collection of test cases, repair process and evaluation,
        after which analysis data is saved and formatted.
        """
        self.start_time = time.time()

        self._collect_test_cases()
        self._repair()
        self._evaluate()        
        self._format_output()

    def _collect_test_cases(self):
        """
        Collects test cases based on parameters.
        """
        if not self.enhance_fault_localization or not self.enhance_validation:
            self._collect_tests4py_tests()
        if self.enhance_fault_localization or self.enhance_validation:
            if self.test_generation_type == TestGenerationType.AVICENNA:
                self._collect_avicenna_tests()
            elif self.test_generation_type == TestGenerationType.GRAMMAR_FUZZER:
                self._collect_fuzzer_tests()
            else:
                raise ValueError("Test generation type not registered.")

        self.collection_duration = time.time() - self.start_time
        LOGGER.info(f"Evaluation pipeline collected \
            {len(self.repair_baseline_failing)} failing and {len(self.repair_baseline_passing)} passing test cases as baseline, \
            {len(self.repair_additional_failing)} failing and {len(self.repair_additional_passing)} passing test cases additionaly."
        )          

    def _collect_tests4py_tests(self):
        """
        Collects test cases from tests4py for baseline.
        """
        report = t4p.checkout(self.subject)
        if report.raised:
            raise report.raised

        systemtests_path = os.path.join("tmp", self.subject.get_identifier(), "tests4py_systemtest_diversity")

        self.repair_baseline_failing = [
            os.path.abspath(os.path.join(systemtests_path, f"failing_test_diversity_{i}")) 
            for i in range(self.num_baseline_failing)
        ]

        self.repair_baseline_passing = [
            os.path.abspath(os.path.join(systemtests_path, f"passing_test_diversity_{i}"))
            for i in range(self.num_baseline_passing)
        ]

    def _collect_avicenna_tests(self):
        """
        Collects test cases, genereated through avicenna.
        """
        report = t4p.checkout(self.subject)
        if report.raised:
            raise report.raised

        dir = os.path.abspath(Path("tmp") / "test_generator" / self.subject.get_identifier())
        saving_path = Path(dir) / "avicenna_test_cases" / f"test_cases_{self.num_additional_failing}"
        if not (self.use_cached_tests and saving_path.exists()):
            self._generate_avicenna_tests(dir, saving_path)

        self.repair_additional_failing = TestGenerator.load_failing_test_paths(saving_path)
        self.repair_additional_passing = TestGenerator.load_passing_test_paths(saving_path)

    def _generate_avicenna_tests(self, dir: Path, saving_path: Path):
        """
        Generates test cases through avicenna.
        """
        identifier: str = f"{self.subject.get_identifier()}_I{str(10)}"
        
        default_param = {
            "max_iterations": 10,
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

        generator.solve_formula(
            max_iterations=self.num_additional_failing,
            negate_formula=False,
            only_unique_inputs=True,
            formula=formula
        )
      
        if self.num_additional_passing > 0:
            generator.solve_formula(
                max_iterations=self.num_additional_passing,
                negate_formula=True,
                only_unique_inputs=True,
                formula=formula
            )

        if saving_path.exists():
            shutil.rmtree(saving_path)

        generator.save_test_cases(saving_path)

    def _collect_fuzzer_tests(self):
        """
        Collect test cases, generated through a grammar based fuzzer.
        """
        report = t4p.checkout(self.subject)
        if report.raised:
            raise report.raised

        dir = os.path.abspath(Path("tmp") / "test_generator" / self.subject.get_identifier())
        saving_path = Path(dir) / "fuzzer_test_cases" / f"test_cases_{self.num_additional_failing}"
        if not (self.use_cached_tests and saving_path.exists()):
            self._generate_fuzzer_tests(dir, saving_path)

        self.repair_additional_failing = TestGenerator.load_failing_test_paths(saving_path)
        self.repair_additional_passing = TestGenerator.load_passing_test_paths(saving_path)

    def _generate_fuzzer_tests(self, dir: Path, saving_path: Path):
        """
        Generates test cases through a grammar based fuzzer.
        """
        generator = GrammarFuzzerTestGenerator(
            oracle = self.benchmark_program.get_oracle(),
            grammar = self.benchmark_program.get_grammar(),
            num_failing = self.num_additional_failing,
            num_passing = self.num_additional_passing,
            out = dir,
            saving_method = "files"
        )

        generator.run()

        if saving_path.exists():
            shutil.rmtree(saving_path)

        generator.save_test_cases(saving_path)

    def _repair(self):
        """
        Starts the repair process.
        """
        baseline_tests = self.repair_baseline_passing + self.repair_baseline_failing
        additional_tests = self.repair_additional_passing + self.repair_additional_failing

        self.fault_localization_test_cases = additional_tests if self.enhance_fault_localization else baseline_tests
        self.validation_test_cases = additional_tests if self.enhance_validation else baseline_tests

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

        self.repair_duration = time.time() - self.start_time - self.collection_duration
        LOGGER.info(f"Evaluation pipeline finished repair with {len(self.patches)} patches.")

    def _evaluate(self):
        """
        Evaluates the patches, generated after the repair, with custom metric
        """      
        dir = Path("evaluation_test_cases")
        fuzzer = GrammarFuzzerTestGenerator(
            oracle = self.benchmark_program.get_oracle(),
            grammar = self.benchmark_program.get_grammar(),
            num_failing = self.num_evaluation_failing,
            num_passing = self.num_evaluation_passing,
            out = dir,
            saving_method = "files"
        )

        if not self.lock_evaluation_test_cases:
            fuzzer.run()

        self.evaluation_passing = TestGenerator.load_passing_test_paths(fuzzer.saving_path)
        self.evaluation_failing = TestGenerator.load_failing_test_paths(fuzzer.saving_path)
        evaluation_tests = self.evaluation_passing + self.evaluation_failing

        LOGGER.info(f"Evaluation pipeline loaded {len(self.evaluation_failing)} failing and {len(self.evaluation_passing)} passing test cases for evaluation.")

        if self.use_parallel_engine:
            engine = Tests4PySystemTestEngine(GenProgFitness(set(self.evaluation_passing), set(self.evaluation_failing)), evaluation_tests, workers=32, out="rep")
        else:
            engine = Tests4PySystemTestSequentialEngine(GenProgFitness(set(self.evaluation_passing), set(self.evaluation_failing)), evaluation_tests, out="rep")
        engine.evaluate(self.patches)

        self.patches.sort(key = lambda patch: patch.fitness, reverse = True)
        self.best_fitness = self.patches[0].fitness if self.patches else 0.0
        if almost_equal(self.best_fitness, 1):
            self.found = True

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

                LOGGER.info(f"Evaluation pipeline evaluated patch {patch}:")
                LOGGER.info(str(get_patch(patch)))
                LOGGER.info(str(matrix))
            else:
                self.patch_matrices[patch] = None
                LOGGER.info(f"Evaluation pipeline could not evaluate {patch}.")

        self.evaluation_duration = time.time() - self.start_time - self.collection_duration - self.repair_duration
        LOGGER.info(f"Evaluation pipeline finished evaluation with best fitness of {self.best_fitness}.")

        

    def _format_output(self):
        """
        Formates output to write in a file.
        """
        output: str = f"APPROACH: {self.approach.__name__}\n"
        output += f"SUBJECT: {self.subject.get_identifier()}\n"

        if self.enhance_fault_localization:
            output += f"Used {len(self.repair_additional_failing)} failing and {len(self.repair_additional_passing)} passing test cases from Avicenna in the fault localization\n"
        else:
            output += f"Used {len(self.repair_baseline_failing)} failing and {len(self.repair_baseline_passing)} passing test cases from Tests4py in the fault localization\n"
        
        if self.enhance_validation:
            output += f"Used {len(self.repair_additional_failing)} failing and {len(self.repair_additional_passing)} passing test cases from Avicenna in the validation\n"
        else:
            output += f"Used {len(self.repair_baseline_failing)} failing and {len(self.repair_baseline_passing)} passing test cases from Tests4py in the validation\n"

        output += f"In total {len(self.fault_localization_test_cases)} for fault localization and {len(self.validation_test_cases)} for validation.\n"
        output += f"The gathering of test cases took {"{:.4f}".format(self.collection_duration)} seconds.\n"
        output += f"The repair ran for {"{:.4f}".format(self.repair_duration)} seconds.\n"
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
            variant = "C" if self.enhance_fault_localization and self.enhance_validation else "F" if self.enhance_fault_localization else "V" if self.enhance_validation else "B"
            best_patch = self.patches[0]
            matrix = self.patch_matrices[best_patch]
            precision = matrix.precision if matrix else None
            recall = matrix.recall if matrix else None
            f1_score = matrix.f1_score if matrix else None
            accuracy = matrix.accuracy if matrix else None
            engine = "parallel" if self.use_parallel_engine else "sequential"
            data = [self.approach.__name__, self.subject.project_name, self.subject.bug_id, 
                    self.fixKit_iterations, engine, variant, 
                    len(self.repair_additional_failing), len(self.repair_additional_passing),
                    len(self.repair_baseline_failing), len(self.repair_baseline_passing),
                    len(self.fault_localization_test_cases), len(self.validation_test_cases),
                    self.collection_duration, self.repair_duration, self.evaluation_duration,
                    self.best_fitness, precision, recall, f1_score, accuracy, len(self.patches)]
            writer.writerow(data)

    @staticmethod
    def write_csv_header(file: str):
        with open(file, "w", newline="") as file:
            writer = csv.writer(file)
            header =["approach", "subject", "bug_id", "iterations", "engine", "variant", "additional_failing", "additional_passing", 
                "baseline_failing", "baseline_passing", "fault_localization_tests", "validation_tests",
                "test_collection_duration", "repair_duration", "evaluation_duration", "best_fitness", 
                "precision", "recall", "f1_score", "accuracy", "patches"]
            writer.writerow(header)        