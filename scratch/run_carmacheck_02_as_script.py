from pathlib import Path

import nbformat


path = Path("carmacheck/02check.ipynb")
nb = nbformat.read(path, as_version=4)

ns = {"__name__": "__main__", "display": print}
for i, cell in enumerate(nb.cells):
    if cell.cell_type != "code":
        continue
    source = "\n".join(
        line for line in cell.source.splitlines()
        if not line.lstrip().startswith("%")
    )
    code = compile(source, f"{path}:cell-{i}", "exec")
    exec(code, ns)

print(f"Executed code cells from {path}")
