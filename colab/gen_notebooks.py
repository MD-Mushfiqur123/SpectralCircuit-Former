"""Build the 9 SpectralCircuit-Former Colab notebooks from nbdefs/* + core.py + nb_extras.py."""
import json
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)

CORE = open(os.path.join(HERE, "core.py")).read().rstrip() + "\nprint('core ready on', DEVICE)\n"
EXTRAS = open(os.path.join(HERE, "nb_extras.py")).read().rstrip() + "\nprint('extras ready')\n"

NEEDS_EXTRAS = {1, 3, 5, 6, 7, 8}

TITLES = {
    1: "01 - Spectral Metrics Laboratory",
    2: "02 - Grokking Phase Transition (Core Finding)",
    3: "03 - Eigenvalue Spectral Density & Marchenko-Pastur Manifolds",
    4: "04 - SGSA Regularizer: tau Ablation",
    5: "05 - Automated Circuit Discovery (Spectral Subgraph Decomposition)",
    6: "06 - Deep Transformers: Rank Collapse & Effective Rank Decay",
    7: "07 - Multi-Task Universal Phase Transition Suite",
    8: "08 - Causal Circuit Surgery (Ablation & Intervention)",
    9: "09 - Scaling, Statistics & Paper-Grade Figures",
}


def strip_extras_refs(text):
    text = text.replace("import nb_extras\n", "")
    text = text.replace("import nb_extras", "")
    return text.replace("nb_extras.", "")


def parse_cells(path):
    raw = open(path).read()
    parts = raw.split("#####CELL#####")
    cells = []
    for part in parts:
        part = part.strip("\n")
        if not part.strip():
            continue
        lines = part.split("\n")
        tag = lines[0].strip()
        body = "\n".join(lines[1:]).strip("\n")
        cells.append((tag, body))
    return cells


def make_cell(kind, text, idx):
    cell = {"cell_type": "markdown" if kind == "md" else "code",
            "metadata": {"id": f"cell-{idx}"},
            "source": text.split("\n")}
    cell["source"] = [ln + "\n" for ln in text.split("\n")]
    if cell["cell_type"] == "code":
        cell["execution_count"] = None
        cell["outputs"] = []
    else:
        cell["source"] = [ln + "\n" for ln in text.split("\n")]
    return cell


def build(nbnum, outdir):
    cells_raw = parse_cells(os.path.join(HERE, "nbdefs", f"nb{nbnum:02d}.py"))
    cells = []
    idx = 0
    intro = ("#!md", (f"# {TITLES[nbnum]}\n\n**SpectralCircuit-Former** research suite — "
                      f"T4 GPU (16GB) এ 100% চালানোর জন্য ডিজাইন করা। "
                      f"সেল উপর থেকে ক্রমানুসারে চালান। GPU: Runtime > Change runtime type > T4 GPU.\n\n"
                      + cells_raw[0][1].split("\n", 1)[1] if cells_raw[0][0] == "#!md" else ""))
    cells.append(make_cell("md", intro[1], idx)); idx += 1
    cells.append(make_cell("code", CORE, idx)); idx += 1
    if nbnum in NEEDS_EXTRAS:
        cells.append(make_cell("code", EXTRAS, idx)); idx += 1
    for tag, body in cells_raw:
        kind = "md" if tag == "#!md" else "code"
        if kind == "code":
            body = strip_extras_refs(body)
        cells.append(make_cell(kind, body, idx)); idx += 1
    cells.append(make_cell("md",
                           "---\n*Run order: উপরের সব code cell উপর থেকে নিচে চালান। "
                           "প্রশ্ন হলে FINDINGS.md দেখুন (repo root)।*",
                           idx))
    nb = {"nbformat": 4, "nbformat_minor": 5,
          "metadata": {"colab": {"provenance": [], "gpuType": "T4"},
                       "kernelspec": {"name": "python3", "display_name": "Python 3"},
                       "language_info": {"name": "python"},
                       "accelerator": "GPU"},
          "cells": cells}
    out = os.path.join(outdir, f"SpectralCircuit-Former_{nbnum:02d}.ipynb")
    with open(out, "w") as f:
        json.dump(nb, f, indent=1, ensure_ascii=False)
    return out


if __name__ == "__main__":
    outdir = os.path.join(HERE, "notebooks")
    os.makedirs(outdir, exist_ok=True)
    for n in range(1, 10):
        path = build(n, outdir)
        nb = json.load(open(path))
        ncode = sum(1 for c in nb["cells"] if c["cell_type"] == "code")
        print(f"built {os.path.basename(path)}  ({len(nb['cells'])} cells, {ncode} code)")
