from pathlib import Path
import tests4py.api as t4p

report = t4p.checkout(t4p.middle_2, Path("tmp"))
if report.raised: raise report.raised

from fixkit.repair.pyae import PyAE
from fixkit.localization.t4p import Tests4PyLocalization
from fixkit.constants import DEFAULT_EXCLUDES

approach = PyAE.from_source(
    Path("tmp", "middle_2"),
    excludes=DEFAULT_EXCLUDES,
    localization=Tests4PyLocalization(
        Path("tmp", "middle_2"),
        events=["line"],
        predicates=["line"],
        metric="Ochiai",
    ),
    k=1,
    is_t4p=True,
    line_mode=True,
)

patches = approach.repair()

patch = patches[0]

patch.mutations

import ast
print(ast.unparse(patch.statements[3]))
print(ast.unparse(patch.statements[1]))

from fixkit.repair.patch import get_patch
print(get_patch(patch))