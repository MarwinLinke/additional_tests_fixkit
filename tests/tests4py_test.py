from isla.solver import ISLaSolver

from pathlib import Path
import tests4py.api as t4p
from subprocess import CompletedProcess

from tests4py.tests.utils import TestResult
from avicenna.avicenna import OracleResult
from fuzzingbook.GrammarFuzzer import GrammarFuzzer

#report = t4p.checkout(t4p.calculator_1, Path("tmp"))
#if report.raised: raise report.raised

# t4p.build(path)

path = Path("tmp", "calculator_1")
grammar = t4p.calculator_1.grammar

fuzzer = GrammarFuzzer(grammar)
passing = []
failing = []



string, test_result = t4p.calculator_1.systemtests.generate_failing_test()
print(string, test_result)

""" result = solver.solve()
print(result)
print(grammar)
run = t4p.run(path, args_or_path=[str(result)],invoke_oracle=True)
oracle = run.test_result

print(run)

print(oracle) """


""" try:
    for i in range(100):
        result = fuzzer.fuzz()
        run = t4p.run(path, args_or_path=[str(result)],invoke_oracle=True)
        oracle = run.test_result
        print(oracle)
        if oracle == TestResult.FAILING:
            failing.append(str(result))
            break
        else:
            passing.append(str(result))
except StopIteration:
    print("stopped") """

print(passing, failing)
