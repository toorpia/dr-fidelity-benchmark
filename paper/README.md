# paper/ — arXiv preprint sources

Frozen snapshot of the benchmark results at release tag **v1.2.0** (REPORT.md is the living document; this paper's numbers stay fixed at that tag).

Build: `pdflatex main && bibtex main && pdflatex main && pdflatex main`

Tables are auto-generated from REPORT.md — never hand-edit values: `python paper/tools/gen_tables_from_report.py` (run from the repo root; regeneration is byte-identical at v1.2.0).

Note: `lmodern` is commented out in `main.tex` for local builds without the package; re-enable it for the arXiv submission (arXiv has it).
