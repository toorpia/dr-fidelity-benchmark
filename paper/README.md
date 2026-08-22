# paper/ — arXiv preprint sources

Frozen snapshot of the benchmark results at release tag **v1.4.0** (REPORT.md is the living document; this paper's numbers stay fixed at that tag).

Build: `pdflatex main && bibtex main && pdflatex main && pdflatex main`

Tables are auto-generated from REPORT.md — never hand-edit values: `python paper/tools/gen_tables_from_report.py` (run from the repo root; the generator takes every row of each table and asserts eight method rows). Figures are verbatim copies of the 13 `figures/<dataset>/*.png` files the text references (kept as copies so the arXiv zip is self-contained).

Submission zip: `main.tex` + `references.bib` + `main.bbl` + `tables/` + `figures/` (no PDF, no aux files). `main.tex` starts with `\pdfoutput=1` so arXiv compiles it with pdflatex.
