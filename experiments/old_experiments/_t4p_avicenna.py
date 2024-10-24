from pathlib import Path
import tests4py.api as t4p

from avicenna.avicenna import OracleResult
from avicenna.avicenna import Avicenna
from typing import List, Tuple
from isla.language import Formula
from isla.language import ISLaUnparser

from tests4py.tests.utils import TestResult
import logging

subject = t4p.pysnooper_1
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
    run = t4p.run(path, args_or_path=str(input).split('\n'),invoke_oracle=True)
    result = run.test_result

    if result == TestResult.FAILING:
        return OracleResult.FAILING
    else:
        return OracleResult.PASSING

inputs = failing + passing
for input in inputs:
    print(input.ljust(30), oracle(input))

avicenna = Avicenna(
    grammar,
    oracle,
    initial_inputs = passing + failing,
    max_iterations = 10
)

diagnosis: Tuple[Formula, float, float] = avicenna.explain()

print("FAILING CONSTRAINTS (DIAGNOSIS):")
print(ISLaUnparser(diagnosis[0]).unparse())
print(f"Avicenna calculated a precision of {diagnosis[1]*100:.2f}% and a recall of {diagnosis[2]*100:.2f}%", end="\n\n")

print("INITIAL INPUTS:\n")

for input in inputs:
    print(input.ljust(30), oracle(input))
    print("----------------------------")

print("\nNEW PASSING INPUTS:")
print(f"Avicenna found {avicenna.get_num_passing_inputs()} new passing inputs, printing first 10\n")

for input in avicenna.report.get_all_passing_inputs()[:10]:
    print(str(input).ljust(30), "PASSING")
    print("----------------------------")

print("\nNEW FAILING INPUTS:")
print(f"Avicenna found {avicenna.get_num_failing_inputs()} new failing inputs, printing first 10\n")

for input in avicenna.report.get_all_failing_inputs()[:10]:
    print(str(input).ljust(30), "FAILING")
    print("----------------------------")