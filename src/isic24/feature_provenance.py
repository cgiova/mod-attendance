"""Where the engineered tabular columns come from, read off the code that creates them.

The provenance of a derived column is a fact about the transformation that made it, and that
transformation is a short sequence of functions in the dataset-build notebook: an assignment per
column, each reading the columns it is computed from.  Written out by hand the map becomes a
second declaration, free to drift from the formulas it describes; parsed from the functions it
cannot.  The sequence matters — a later function may reassign a column an earlier one wrote, and
then the stored values are the later formula's — so the assignments are walked in order and a
column's sources are resolved against the state reached at that point.

`verify_engineered` runs the same functions again and compares their output with the columns
already on the frame, so the map is held against the values on disk and not only against the
names.  It feeds them EXACTLY the columns the map names, which is what makes an incomplete map
raise instead of agreeing.

The comparison belongs on the metadata as the build wrote it, **before any imputation**: training
fills missing values in raw and engineered columns independently, so on an imputed frame the
identity can break on the filled rows without anything being wrong.
"""

from __future__ import annotations

import ast
import json

import numpy as np
import pandas as pd


def _function_source(nb_path, func_name):
    """The source of the one function of that name defined in the notebook."""
    found = []
    for cell in json.load(open(nb_path))["cells"]:
        if cell["cell_type"] != "code":
            continue
        src = "".join(cell["source"])
        try:
            tree = ast.parse(src)
        except SyntaxError:
            continue
        found += [ast.get_source_segment(src, n) for n in tree.body
                  if isinstance(n, ast.FunctionDef) and n.name == func_name]
    if len(found) != 1:
        raise ValueError(f"{nb_path}: {len(found)} cells define {func_name}(), exactly one is "
                         f"needed for the provenance to have a single source")
    return found[0]


def _column(node):
    """The column name a `frame[\"literal\"]` node addresses, or None."""
    if (isinstance(node, ast.Subscript) and isinstance(node.value, ast.Name)
            and isinstance(node.slice, ast.Constant) and isinstance(node.slice.value, str)):
        return node.slice.value
    return None


def engineered_provenance(nb_path, func_names=("feature_engineering",)):
    """The source of the functions that build the engineered columns, and what each one reads.

    Returns `(source, roots)`, where `source` is the functions joined in the order given and
    `roots[col]` holds the columns `col` ultimately comes from: a column assigned from another
    assigned column resolves through it, so every entry is expressed in columns the build never
    writes.
    """
    sources = [_function_source(nb_path, f) for f in func_names]
    roots = {}
    for source in sources:
        for st in ast.parse(source).body[0].body:
            if not (isinstance(st, ast.Assign) and len(st.targets) == 1):
                continue
            col = _column(st.targets[0])
            if col is None:
                continue
            reads = {c for c in map(_column, ast.walk(st.value)) if c is not None}
            roots[col] = set().union(*(roots.get(r, {r}) for r in reads)) if reads else set()
    return "\n\n".join(sources), roots


def verify_engineered(df, source, roots, func_names=("feature_engineering",), rtol=1e-9):
    """Rows on which the column carried by `df` differs from what the build recomputes, per column.

    The map says which columns an engineered one reads; this says the values on disk are what
    reading them produces.  The functions are handed the root columns and nothing else, so a map
    that names too few of them fails on the selection instead of quietly agreeing.
    """
    ns = {"np": np, "pd": pd}
    exec(source, ns)
    got = df[sorted(set().union(*roots.values()))].copy()
    for func_name in func_names:
        out = ns[func_name](got)
        got = out[0] if isinstance(out, tuple) else out
    differences = {}
    for col in roots:
        new, old = got[col], df[col]
        if pd.api.types.is_numeric_dtype(old):
            differs = ~np.isclose(new.to_numpy("float64"), old.to_numpy("float64"),
                                  rtol=rtol, atol=0.0, equal_nan=True)
        else:
            differs = (new != old).to_numpy() & ~(new.isna() & old.isna()).to_numpy()
        differences[col] = int(np.count_nonzero(differs))
    return pd.Series(differences, dtype="int64")
