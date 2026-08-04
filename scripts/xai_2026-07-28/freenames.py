import ast, sys, builtins
BUILT=set(dir(builtins))
class Scope:
    def __init__(s,parent=None): s.names=set(); s.parent=parent
    def has(s,n): return n in s.names or (s.parent.has(n) if s.parent else False)
class V(ast.NodeVisitor):
    def __init__(s): s.mod=Scope(); s.free=set(); s.defined=set()
    def run(s,tree):
        # pass 1: module-level bindings
        for n in ast.walk(tree):
            if isinstance(n,(ast.FunctionDef,ast.AsyncFunctionDef,ast.ClassDef)): s.mod.names.add(n.name)
            elif isinstance(n,ast.Name) and isinstance(n.ctx,ast.Store): s.mod.names.add(n.id)
            elif isinstance(n,(ast.Import,ast.ImportFrom)):
                for a in n.names: s.mod.names.add((a.asname or a.name).split(".")[0])
            elif isinstance(n,ast.arg): s.mod.names.add(n.arg)
            elif isinstance(n,ast.ExceptHandler) and n.name: s.mod.names.add(n.name)
        s.defined=set(s.mod.names)
        for n in ast.walk(tree):
            if isinstance(n,ast.Name) and isinstance(n.ctx,ast.Load) and n.id not in s.mod.names and n.id not in BUILT:
                s.free.add(n.id)
        return s
for f in sys.argv[1:]:
    src=open(f).read()
    # strip ipython magics/shell
    src="\n".join(("#"+l if l.lstrip().startswith(("!","%")) else l) for l in src.splitlines())
    try: t=ast.parse(src)
    except SyntaxError as e: print(f"{f}: SYNTAX {e}"); continue
    v=V().run(t)
    print(f"\n=== {f.split('/')[-1]} free names ({len(v.free)}) ===")
    print("  "+", ".join(sorted(v.free)))
