import random
import time
import csv
import os
import numpy as np
from pathlib import Path
from typing import Type, Dict, Any, Tuple, List

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
from fixkit.logger import LOGGER

from repair_evaluation_matrix import RepairEvaluationMatrix
from data import almost_equal

class EvaluationPipeline():

    def __init__(
        self,
        approach: Type[GeneticRepair], 
        parameters: Dict[str, Any], 
        subject: Project,
        repair_iterations: int,
        seed: int,
        repair_tests_path: str,
        evaluation_tests_path: str,
        num_baseline_failing: int = 1,
        num_baseline_passing: int = 10,
        num_additional_failing: int = 50,
        num_additional_passing: int = 50,
        num_evaluation_failing: int = 50,
        num_evaluation_passing: int = 50,        
        enhance_localization: bool = False,
        enhance_validation: bool = False,
        use_parallel_engine: bool = False
    ):
        """
        :param Type[GeneticRepair] approach: The used repair approach.
        :param Project subject: The tests4py subject.
        :param int repair_iterations: The number of iterations the repair will run for.
        :param int seed: The seed used for random and numpy.random.
        :param int num_baseline_failing: The number of failing test cases used for the baseline.
        :param int num_baseline_passing: The number of passing test cases used for the baseline.
        :param int num_additional_failing: The number of additional failing test cases used.
        :param int num_additional_passing: The number of additional passing test cases used.
        :param int num_evaluation_failing: The number of failing test cases used for evaluating the quality of patches.
        :param int num_evaluation_passing: The number of passing test cases used for evaluating the quality of patches.
        :param bool enhance_fault_localization: If this is true, additional test cases will be used in the fault localization.
        :param bool enhance_validation: If this is true, additional test cases will be used in the validation.
        :param bool use_parallel_engine: Uses either the parallel or sequential engine for evaluating the patches.
        """
        self.approach: Type[GeneticRepair] = approach
        self.parameters: Dict[str, Any] = parameters
        self.subject: Project = subject
        self.fixKit_iterations = repair_iterations

        self.repair_tests_path = repair_tests_path
        self.evaluation_tests_path = evaluation_tests_path

        self.num_baseline_failing = num_baseline_failing
        self.num_baseline_passing = num_baseline_passing
        self.num_additional_failing = num_additional_failing
        self.num_additional_passing = num_additional_passing
        self.num_evaluation_failing = num_evaluation_failing
        self.num_evaluation_passing = num_evaluation_passing

        self.enhance_localization = enhance_localization
        self.enhance_validation = enhance_validation
        self.use_parallel_engine = use_parallel_engine

        self.repair_additional_failing: List[str] = []
        self.repair_additional_passing: List[str] = []

        self.repair_baseline_failing: List[str] = []
        self.repair_baseline_passing: List[str] = []

        self.evaluation_failing: List[str] = []
        self.evaluation_passing: List[str] = []

        self.seed = seed
        self.patches: List[GeneticCandidate] = []
        self.matrices: List[RepairEvaluationMatrix] = []
        self.filtered_patches: List[GeneticCandidate] = []
        self.num_equivalent_patches: Dict[GeneticCandidate, int] = {}
        self.found: bool = False
        self.best_fitness: float = 0.0
        self.best_f1_score: float = 0.0

        self.start_time = 0.0
        self.collection_duration = 0.0
        self.repair_duration = 0.0
        self.evaluation_duration = 0.0

        self.num_generations = 0

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
        report = t4p.checkout(self.subject)
        if report.raised:
            raise report.raised

        if not self.enhance_localization or not self.enhance_validation:
            self._collect_tests4py_tests()
        if self.enhance_localization or self.enhance_validation:
            self._collect_fuzzer_tests()

        self.collection_duration = time.time() - self.start_time
        LOGGER.info(f"Evaluation pipeline collected "
                    f"{len(self.repair_baseline_failing)} failing "
                    f"and {len(self.repair_baseline_passing)} passing test cases as baseline, "
                    f"{len(self.repair_additional_failing)} failing "
                    f"and {len(self.repair_additional_passing)} passing test cases additionally.")          

    def _collect_tests4py_tests(self):
        """
        Collects test cases from tests4py for baseline.
        """
        systemtests_path = os.path.join("tmp", self.subject.get_identifier(), "tests4py_systemtest_diversity")

        self.repair_baseline_failing = [
            os.path.abspath(os.path.join(systemtests_path, f"failing_test_diversity_{i}")) 
            for i in range(self.num_baseline_failing)
        ]

        self.repair_baseline_passing = [
            os.path.abspath(os.path.join(systemtests_path, f"passing_test_diversity_{i}"))
            for i in range(self.num_baseline_passing)
        ]

    def _collect_fuzzer_tests(self):
        """
        Collect test cases, generated through a grammar based fuzzer.
        """
        self.repair_additional_failing = TestGenerator.load_failing_test_paths(
            self.repair_tests_path, self.num_additional_failing
        )
        
        self.repair_additional_passing = TestGenerator.load_passing_test_paths(
            self.repair_tests_path, self.num_additional_passing
        )

    def _repair(self):
        """
        Starts the repair process.
        """
        self.localization_failing = (
            self.repair_additional_failing 
            if self.enhance_localization 
            else self.repair_baseline_failing)
        self.localization_passing = (
            self.repair_additional_passing 
            if self.enhance_localization 
            else self.repair_baseline_passing)    
        self.localization_tests = self.localization_failing + self.localization_passing

        self.validation_failing = (
            self.repair_additional_failing 
            if self.enhance_validation 
            else self.repair_baseline_failing)
        self.validation_passing = (
            self.repair_additional_passing 
            if self.enhance_validation 
            else self.repair_baseline_passing)  
        self.validation_tests = self.validation_failing + self.validation_passing

        params = self.parameters
        if self.approach != PyAE:
            params["max_generations"] = self.fixKit_iterations

        # Special pysnooper excludes because they caused errors when mutating.
        pysnooper_excludes = ["pysnooper/__init__.py", "pysnooper/pycompat.py"]

        approach = self.approach.from_source(
            src=Path("tmp", self.subject.get_identifier()),
            excludes=DEFAULT_EXCLUDES + pysnooper_excludes,
            localization=Tests4PySystemtestsLocalization(
                src=Path("tmp", self.subject.get_identifier()),
                events=["line"],
                predicates=["line"],
                metric="Ochiai",
                out="rep",
                tests = self.localization_tests,
                excluded_files=DEFAULT_EXCLUDES + pysnooper_excludes
            ),
            out="rep",
            is_t4p=True,
            is_system_test = True,
            failing_tests = self.validation_failing,
            passing_tests = self.validation_passing,
            serial = False,
            **params
        )

        self._set_seed()
        self.patches: List[GeneticCandidate] = approach.repair()
        self.num_generations = approach.current_gen

        LOGGER.info(f"Evaluation pipeline generates random number for seed verification: {random.randint(0, 1000)}, {np.random.randint(0, 1000)}.")
        LOGGER.info(f"Evaluation pipeline finished repair with {len(self.patches)} patches.")

        self.repair_duration = time.time() - self.start_time - self.collection_duration


    def _filter_patches(self):
        for patch in self.patches:
            if patch in self.filtered_patches:
                self.num_equivalent_patches[patch] += 1
            else:
                self.filtered_patches.append(patch)
                self.num_equivalent_patches[patch] = 1

    def _evaluate(self):
        """
        Evaluates the patches, generated after the repair, with custom metric.
        """      
        self.evaluation_failing = TestGenerator.load_failing_test_paths(
            self.evaluation_tests_path, self.num_evaluation_failing
        )
        self.evaluation_passing = TestGenerator.load_passing_test_paths(
            self.evaluation_tests_path, self.num_evaluation_passing
        )
        evaluation_tests = self.evaluation_passing + self.evaluation_failing

        LOGGER.info(f"Evaluation pipeline loaded {len(self.evaluation_failing)} failing and {len(self.evaluation_passing)} passing test cases for evaluation.")

        fitness = GenProgFitness(set(self.evaluation_passing), set(self.evaluation_failing))
        if self.use_parallel_engine:
            engine = Tests4PySystemTestEngine(fitness, evaluation_tests, workers=32, out="rep")
        else:
            engine = Tests4PySystemTestSequentialEngine(fitness, evaluation_tests, out="rep")
        engine.evaluate(self.patches)

        self._filter_patches()

        patches: List[GeneticCandidate] = self.filtered_patches

        patches.sort(key = lambda patch: patch.fitness, reverse = True)
        self.best_fitness = patches[0].fitness if patches else 0.0
        if almost_equal(self.best_fitness, 1):
            self.found = True

        for i in range(len(patches)):
            patch = patches[i]
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
                self.matrices.append(matrix)
                
                self.best_f1_score = max(self.best_f1_score, matrix.f1_score)

                LOGGER.info(f"Evaluation pipeline evaluated patch {patch}:\n"
                            f"Mutations: {patch.mutations}\n"
                            f"{get_patch(patch)}\n"
                            f"{matrix}\n")
            else:
                self.matrices.append(None)
                LOGGER.info(f"Evaluation pipeline could not evaluate {patch}.")

        self.evaluation_duration = time.time() - self.start_time - self.collection_duration - self.repair_duration
        LOGGER.info(f"Evaluation pipeline finished evaluation with best fitness of {self.best_fitness}.")        

    def _format_output(self):
        """
        Formates output to write in a file.
        """
        localization_tests = (
            f"{len(self.repair_additional_failing)} failing, {len(self.repair_additional_passing)} passing"
            if self.enhance_localization
            else f"{len(self.repair_baseline_failing)} failing, {len(self.repair_baseline_passing)} passing"
        )

        validation_tests = (
            f"{len(self.repair_additional_failing)} failing, {len(self.repair_additional_passing)} passing"
            if self.enhance_validation
            else f"{len(self.repair_baseline_failing)} failing, {len(self.repair_baseline_passing)} passing"
        )

        enhanced_localization = "(Enhanced)" if self.enhance_localization else "(Baseline)"
        enhanced_validation = "(Enhanced)" if self.enhance_validation else "(Baseline)"

        output = (
            f"APPROACH: {self.approach.__name__}\n"
            f"SUBJECT: {self.subject.get_identifier()}\n\n"
            f"Test Cases Used:\n"
            f"  - {enhanced_localization} Fault Localization: {localization_tests} "
            f"(Total: {len(self.localization_tests)})\n"
            f"  - {enhanced_validation} Validation: {validation_tests} "
            f"(Total: {len(self.validation_tests)})\n\n"
            f"Execution Times:\n"
            f"  - Test Case Gathering: {self.collection_duration:.4f} seconds\n"
            f"  - Repair: {self.repair_duration:.4f} seconds\n"
            f"  - Evaluation: {self.evaluation_duration:.4f} seconds\n\n"
            f"Results:\n"
            f"  - Valid Patch Found: {self.found}\n"
            f"  - Best Fitness: {self.best_fitness:.4f}\n"
            f"  - Best F1 Score: {self.best_f1_score:.4f}\n"
            f"  - Total Patches Found: {len(self.patches)}\n"
            f"  - Generations Completed: {self.num_generations}/{self.fixKit_iterations}\n\n"
        )    

        output += "".ljust(100, "%") + "\n\nSorted patches by descending fitness:\n\n"

        for i, patch in enumerate(self.filtered_patches):
            output += (
                f"{patch}\n"
                f"Found {self.num_equivalent_patches[patch]} equivalent patches.\n"
                f"Mutations: {patch.mutations}\n"
            )
            matrix = self.matrices[i] or "\nNo tests4py report was found, matrix could not be calculated."
            output += str(matrix)
            fix = get_patch(patch)
            if not fix:
                output += "\nPatch could not be printed.\n\n"
            else:
                output += f"\n{fix}\n"
            output += "".ljust(100, "_") + "\n\n"

        self.output = output

    def write_report(self, path: os.PathLike):
        Path(path).mkdir(parents=True, exist_ok=True)
        variant = "C" if self.enhance_localization and self.enhance_validation else "F" if self.enhance_localization else "V" if self.enhance_validation else "B"
        tests_identifier = (f"{self.num_additional_failing}-{self.num_additional_passing}" if variant != "B"
                            else f"{self.num_baseline_failing}-{self.num_baseline_passing}")
        engine_identifier = "P" if self.use_parallel_engine else "S"
        file_name = (f"{self.approach.__name__}_{self.subject.project_name}{self.subject.bug_id}"
                     f"_I{self.fixKit_iterations}_{variant}_({tests_identifier})"
                     f"_{engine_identifier}_{self.seed}")

        out = Path(path) / f"{file_name}.txt"
        with open(out, "w") as f:
            f.write(self.output)

    def write_to_csv(self, directory: os.PathLike, file_name: str):
        path = Path(directory)
        path.mkdir(parents=True, exist_ok=True)
        with open(path / file_name, "a", newline="") as file:
            writer = csv.writer(file)
            variant = "C" if self.enhance_localization and self.enhance_validation else "F" if self.enhance_localization else "V" if self.enhance_validation else "B"
            matrix = self.matrices[0]
            precision = matrix.precision if matrix else None
            recall = matrix.recall if matrix else None
            f1_score = matrix.f1_score if matrix else None
            accuracy = matrix.accuracy if matrix else None
            engine = "parallel" if self.use_parallel_engine else "sequential"
            data = [self.approach.__name__, self.subject.project_name, self.subject.bug_id, 
                    self.fixKit_iterations, engine, variant, 
                    len(self.repair_additional_failing), len(self.repair_additional_passing),
                    len(self.repair_baseline_failing), len(self.repair_baseline_passing),
                    len(self.localization_tests), len(self.validation_tests),
                    self.collection_duration, self.repair_duration, self.evaluation_duration,
                    self.best_fitness, precision, recall, f1_score, accuracy, len(self.patches),
                    self.num_generations]
            writer.writerow(data)

    @staticmethod
    def write_csv_header(directory: os.PathLike, file_name: str):
        path = Path(directory)
        path.mkdir(parents=True, exist_ok=True)
        with open(path / file_name, "w", newline="") as file:
            writer = csv.writer(file)
            header =["approach", "subject", "bug_id", "iterations", "engine", "variant", "additional_failing", "additional_passing", 
                "baseline_failing", "baseline_passing", "fault_localization_tests", "validation_tests",
                "test_collection_duration", "repair_duration", "evaluation_duration", "best_fitness", 
                "precision", "recall", "f1_score", "accuracy", "patches", "generations"]
            writer.writerow(header)       