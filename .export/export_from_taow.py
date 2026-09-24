#!/usr/bin/env python3
"""Export workshop notebooks from taow/workspace into this public repo.

Deterministic: same inputs give the same outputs. Run from anywhere:

    ../taow/.venv/bin/python .export/export_from_taow.py   # needs nbconvert

What it does to each notebook:
- keeps the public filename, so existing Colab links keep working
- adds the Open-in-Colab badge as the first cell
- rewrites relative image paths to raw.githubusercontent URLs (Colab cannot
  resolve relative paths) and copies those assets into the repo
- rewrites links to sibling taow notebooks to their public Colab links
- replaces the footer with the repo's CC BY footer
- writes the rendered Markdown copy next to it
"""
import json
import re
import shutil
import subprocess
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
TAOW = REPO.parent / "taow" / "workspace"
RAW = "https://raw.githubusercontent.com/techaarvam/byom_workshop/main/"
COLAB = "https://colab.research.google.com/github/techaarvam/byom_workshop/blob/main/"

# taow source name -> public name
NOTEBOOKS = {
    "08_attention_ann.ipynb": "attention_ann.ipynb",
    "09_position_embedding_residuals_layers.ipynb": "attention_ann_part2_puzzle_solution.ipynb",
}

FOOTER = """---

<img src="assets/techaarvam_logo.png" alt="Tech Aarvam - Dream Build Inspire" width="140">

### About this file

This notebook is part of the support files for the TechAarvam workshop
**[Build Your Own Model](https://www.techaarvam.com/workshops/build-your-own-model)**.

- Website: <https://www.techaarvam.com>
- Workshop files repository: <https://github.com/techaarvam/byom_workshop>
- YouTube: <https://www.youtube.com/@TechAarvam>

© TechAarvam. You are free to use, copy, modify, share and build on this
material, including for commercial purposes, **provided you credit
TechAarvam** and link back to <https://www.techaarvam.com>. Please keep
this notice with any copy or derivative. Provided as-is, without warranty."""

REF = re.compile(r'(src="|\]\()([^)"\s]+)')


def to_lines(text):
    lines = text.split("\n")
    return [l + "\n" for l in lines[:-1]] + [lines[-1]]


def rewrite(src, assets):
    # Link text that names a taow notebook names its public copy instead.
    for old, new in NOTEBOOKS.items():
        src = src.replace(f"[{old}](", f"[{new}](")

    def sub(m):
        prefix, ref = m.groups()
        if ref.startswith(("http", "#", "mailto:")):
            return m.group(0)
        if ref in NOTEBOOKS:
            return prefix + COLAB + NOTEBOOKS[ref]
        if ref.startswith("assets/"):
            assets.add(ref)
            return prefix + RAW + ref
        raise SystemExit(f"unhandled relative reference: {ref}")
    return REF.sub(sub, src)


def export(src_name, dst_name):
    nb = json.loads((TAOW / src_name).read_text())
    cells = nb["cells"]
    last = "".join(cells[-1]["source"])
    if not last.lstrip().startswith("---"):
        raise SystemExit(f"{src_name}: last cell is not the footer")
    cells[-1]["source"] = to_lines(FOOTER)
    # The public repo is not a numbered series: "# 08 - Title" -> "# Title".
    first = "".join(cells[0]["source"])
    cells[0]["source"] = to_lines(re.sub(r"^# \d+[a-z]? - ", "# ", first))
    assets = set()
    for c in cells:
        if c["cell_type"] == "markdown":
            c["source"] = to_lines(rewrite("".join(c["source"]), assets))
    badge = (f"[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)]"
             f"({COLAB}{dst_name})\n\n"
             "*Part of the TechAarvam workshop support files — "
             "[Build Your Own Model](https://www.techaarvam.com/workshops/build-your-own-model).*")
    cells.insert(0, {"cell_type": "markdown", "metadata": {}, "source": to_lines(badge)})
    nb["metadata"].setdefault("colab", {"provenance": []})
    out = REPO / dst_name
    out.write_text(json.dumps(nb, indent=1, ensure_ascii=False) + "\n")
    for a in sorted(assets):
        (REPO / a).parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(TAOW / a, REPO / a)
    subprocess.run([sys.executable, "-m", "nbconvert", "--to", "markdown", "--log-level", "WARN",
                    "--output", out.stem, str(out)], check=True, cwd=REPO)
    print(f"{src_name} -> {dst_name}: {len(cells)} cells, {len(assets)} assets")


if __name__ == "__main__":
    for s, d in NOTEBOOKS.items():
        export(s, d)
