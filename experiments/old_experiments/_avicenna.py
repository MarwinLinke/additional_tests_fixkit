import math
import string
from avicenna.avicenna import OracleResult, Avicenna
from typing import List, Tuple
from isla.language import Formula
from isla.language import ISLaUnparser

def calculator(inp: str) -> float:
    return eval(
        str(inp), {"sqrt": math.sqrt, "sin": math.sin, "cos": math.cos, "tan": math.tan}
    )


def oracle(inp: str):
    try:
        calculator(inp)
    except ValueError as e:
        return OracleResult.FAILING
    return OracleResult.PASSING

initial_inputs = ['sqrt(-3)', 'cos(1)']

for inp in initial_inputs:
    print(inp.ljust(30), oracle(inp))


grammar = {
    "<start>": ["<arith_expr>"],
    "<arith_expr>": ["<function>(<number>)"],
    "<function>": ["sin", "cos", "tan", "sqrt"],
    "<number>": ["<maybe_minus><onenine><maybe_digits><maybe_frac>"],
    "<maybe_minus>": ["", "-"],
    "<onenine>": [str(num) for num in range(1, 10)],
    "<digit>": list(string.digits),
    "<maybe_digits>": ["", "<digits>"],
    "<digits>": ["<digit>", "<digit><digits>"],
    "<maybe_frac>": ["", ".<digits>"],
}

avicenna = Avicenna(
    grammar,
    oracle,
    initial_inputs,
)

diagnosis: Tuple[Formula, float, float] = avicenna.explain()


print(f"Avicenna determined the following constraints to describe the failure circumstances:\n")
print(ISLaUnparser(diagnosis[0]).unparse())
print(f"Avicenna calculated a precision of {diagnosis[1]*100:.2f}% and a recall of {diagnosis[2]*100:.2f}%", end="\n\n")