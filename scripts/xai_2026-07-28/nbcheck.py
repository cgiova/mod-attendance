"""Linear-execution free-name checker for a notebook.

Walks cells in order, accumulating module-level bindings, and reports any Load of a name
that has not been bound by an earlier cell (or this one) and is not a builtin.  Catches
the definition-after-use class of bug -- e.g. the post-hoc notebook calling
`encoded_indices_for_raw` at cell 24 while defining it at cell 38.
"""
import ast
import builtins
import json
import sys

BUILT = set(dir(builtins)) | {"__name__", "__file__", "get_ipython", "display", "In", "Out"}


def bindings(tree):
    out = set()
    for n in ast.walk(tree):
        if isinstance(n, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)):
            out.add(n.name)
        elif isinstance(n, ast.Name) and isinstance(n.ctx, ast.Store):
            out.add(n.id)
        elif isinstance(n, (ast.Import, ast.ImportFrom)):
            for a in n.names:
                out.add((a.asname or a.name).split(".")[0])
        elif isinstance(n, ast.arg):
            out.add(n.arg)
        elif isinstance(n, ast.ExceptHandler) and n.name:
            out.add(n.name)
        elif isinstance(n, ast.alias) and n.asname:
            out.add(n.asname)
        elif isinstance(n, (ast.Global, ast.Nonlocal)):
            out.update(n.names)
    return out


def loads(tree):
    return {n.id for n in ast.walk(tree)
            if isinstance(n, ast.Name) and isinstance(n.ctx, ast.Load)}


def check(path):
    nb = json.load(open(path))
    known, problems = set(), []
    for i, c in enumerate(nb["cells"]):
        if c["cell_type"] != "code":
            continue
        src = "".join(c["source"])
        src = "\n".join(("#" + l if l.lstrip().startswith(("!", "%")) else l)
                        for l in src.splitlines())
        try:
            tree = ast.parse(src)
        except SyntaxError as e:
            problems.append((i, f"SYNTAX: {e}"))
            continue
        b = bindings(tree)
        for name in sorted(loads(tree) - b - known - BUILT):
            problems.append((i, name))
        known |= b
    return problems


if __name__ == "__main__":
    for p in sys.argv[1:]:
        pr = check(p)
        print(f"\n=== {p.split('/')[-1]} ===")
        if not pr:
            print("  no undefined names")
        for i, n in pr:
            print(f"  cell {i:3d}: {n}")
