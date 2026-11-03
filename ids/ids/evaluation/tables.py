"""Write results as CSV (for analysis) and LaTeX (for paper/tables/)."""

from __future__ import annotations

import csv
from typing import Dict

from .metrics import ClassificationReport

_COLS = ["precision", "recall", "f1", "fpr", "attribution_acc"]


def write_csv(path: str, systems: Dict[str, ClassificationReport]) -> None:
    with open(path, "w", newline="") as fh:
        w = csv.writer(fh)
        w.writerow(["system", "family"] + _COLS)
        for name, rep in systems.items():
            w.writerow([name, "ALL"] + [f"{rep.row()[c]:.4f}" for c in _COLS])
            for fam, sub in sorted(rep.per_family.items()):
                w.writerow([name, fam] + [f"{sub.row()[c]:.4f}" for c in _COLS])


def write_latex(path: str, systems: Dict[str, ClassificationReport], caption: str, label: str) -> None:
    lines = [
        r"\begin{table}[t]", r"\centering", rf"\caption{{{caption}}}\label{{{label}}}",
        r"\begin{tabular}{lrrrrr}\toprule",
        r"System & Prec. & Rec. & F1 & FPR & Attr.\ acc. \\\midrule",
    ]
    for name, rep in systems.items():
        r = rep.row()
        lines.append(
            f"{name} & {r['precision']:.3f} & {r['recall']:.3f} & {r['f1']:.3f} & {r['fpr']:.4f} & {r['attribution_acc']:.3f} \\\\"
        )
    lines += [r"\bottomrule", r"\end{tabular}", r"\end{table}", ""]
    with open(path, "w") as fh:
        fh.write("\n".join(lines))
