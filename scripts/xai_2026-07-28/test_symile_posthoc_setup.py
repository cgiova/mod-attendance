"""Check post-hoc cell dependencies without executing producers or writing artifacts."""

import builtins
import json
from pathlib import Path
import re
import symtable


NOTEBOOK = Path(__file__).resolve().parents[2] / (
    "notebooks/modality_attendance/symile_modattendance_posthoc.ipynb"
)
PREDECESSORS = {
    "@@GAMMA_ATTENDANCE@@": "@@LOGIT_DECOMP@@",
    "@@STRATA_SECTION@@ 2/3": "@@STRATA_SECTION@@ 1/3",
    "@@STRATA_SECTION@@ 3/3": "@@STRATA_SECTION@@ 1/3",
}


def global_reads(table):
    names = {s.get_name() for s in table.get_symbols()
             if s.is_global() and s.is_referenced()}
    for child in table.get_children():
        names.update(global_reads(child))
    return names


def test_posthoc_setup():
    notebook = json.loads(NOTEBOOK.read_text())
    setup = set(dir(builtins)) | {"display", "get_ipython"}
    definitions = {}
    in_setup = True
    for cell in notebook["cells"]:
        if cell["cell_type"] != "code":
            continue
        source = "".join(cell["source"])
        marker = re.search(r"@@[A-Z0-9_]+@@(?: [123]/3)?", source.splitlines()[0]).group()
        compile(source, str(NOTEBOOK), "exec")
        table = symtable.symtable(source, str(NOTEBOOK), "exec")
        defined = {s.get_name() for s in table.get_symbols()
                   if s.is_assigned() or s.is_imported() or s.is_namespace()}
        available = setup | defined
        if marker in PREDECESSORS:
            available |= definitions[PREDECESSORS[marker]]
        missing = global_reads(table) - available
        assert not missing, f"{marker}: names unavailable after setup: {sorted(missing)}"
        definitions[marker] = defined
        if marker == "@@STOP@@":
            in_setup = False
        if in_setup:
            setup.update(defined)
    assert not in_setup, "The notebook must declare the setup boundary with @@STOP@@"


if __name__ == "__main__":
    test_posthoc_setup()
    print("PASS: setup supplies producer dependencies; only declared logical groups share state.")
