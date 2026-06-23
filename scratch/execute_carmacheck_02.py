from pathlib import Path

import nbformat
from nbclient import NotebookClient


path = Path("carmacheck/02check.ipynb")
nb = nbformat.read(path, as_version=4)

client = NotebookClient(
    nb,
    timeout=1200,
    kernel_name="python3",
    resources={"metadata": {"path": str(path.parent)}},
)

client.execute()
nbformat.write(nb, path)
print(f"Executed {path}")
