from pathlib import Path
import tests4py.api as t4p
from fixkit.repair.pyae import PyAE
from fixkit.localization.t4p import Tests4PyLocalization
from fixkit.constants import DEFAULT_EXCLUDES

from isla.solver import ISLaSolver
import string

from avicenna.avicenna import OracleResult
from avicenna.avicenna import Avicenna
from typing import List, Tuple
from isla.language import Formula
from isla.language import ISLaUnparser

from tests4py.tests.utils import TestResult
import logging



subject = t4p.cookiecutter_2
passing_test, _ = subject.systemtests.generate_passing_test()
failing_test, _ = subject.systemtests.generate_failing_test()
passing = [passing_test]
failing = [failing_test]

grammar = subject.grammar
t4p_oracle = subject.api.oracle

report = t4p.checkout(subject, Path("tmp"))
if report.raised: raise report.raised
path = Path("tmp", subject.get_identifier())

print(grammar)

t4p.build()

def oracle(input: str) -> OracleResult:
    # print(input)
    run = t4p.run(path, args_or_path=str(input).split('\n'),invoke_oracle=True)
    result = run.test_result

    if result == TestResult.FAILING:
        return OracleResult.FAILING
    else:
        return OracleResult.PASSING

inputs = failing + passing
for input in inputs:
    print(input.ljust(30), oracle(input))


print("INIT")


avicenna = Avicenna(
    grammar,
    oracle,
    initial_inputs = passing + failing,
    max_iterations = 2
)

LOGGER = logging.getLogger("tests4py")
LOGGER.disabled = True

print("START DIAGNOSING")

diagnosis: Tuple[Formula, float, float] = avicenna.explain()



print(f"Avicenna determined the following constraints to describe the failure circumstances:\n")
print(ISLaUnparser(diagnosis[0]).unparse())
print(f"Avicenna calculated a precision of {diagnosis[1]*100:.2f}% and a recall of {diagnosis[2]*100:.2f}%", end="\n\n")

print(inputs)

failures = avicenna.report.get_all_failing_inputs()
print(f"INPUTS: {[str(failure) for failure in failures]}")
print(f"Number of failing inputs: {avicenna.get_num_failing_inputs()}")


""" solver = ISLaSolver(
        grammar,
        formula=diagnosis[0],
        enable_optimized_z3_queries=False)

for _ in range(20):
        try:
            inp = solver.solve()
            print(str(inp).ljust(30), oracle(inp))
        except StopIteration:
            continue """

""" for diagnosis in avicenna.get_equivalent_best_formulas():
    solver = ISLaSolver(
        grammar,
        formula=diagnosis[0],
        enable_optimized_z3_queries=False)
    
    for _ in range(20):
        try:
            inp = solver.solve()
            print(str(inp).ljust(30), oracle(inp))
        except StopIteration:
            continue
 """