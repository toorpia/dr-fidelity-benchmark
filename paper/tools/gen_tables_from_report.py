#!/usr/bin/env python3
"""Generate paper/tables/*.tex verbatim from REPORT.md (single source of truth).

Usage (from repo root):
    python paper/tools/gen_tables_from_report.py [--report REPORT.md] [--outdir paper/tables]

Never hand-edit the generated .tex files; if a value changes, re-run the
benchmark, regenerate REPORT.md, then re-run this script. Markdown emphasis is
mapped as: **bold** = best in column, *italic* = worst, "⚠" = outright-failure
flag -> \\textsuperscript{\\dag} (bold is stripped on flagged cells so that
"best" and "failure" stay visually distinct).
"""
import argparse
import os
import re

COL_HDR_MAP = {
    'method': 'method', 'full pts': 'full', 'near pts': 'near',
    'outlier pts': 'outl.', 'Σ': r'$\Sigma$',
    'full · global': r'full $\rho$ (global)',
    'near · first-mode band': r'near $\rho$ (first-mode)',
    'tight-cluster scale ×': r'scale $\times$', 'recall@15': 'recall@15',
    'trust@15': 'trust@15', 'continuity@15': 'cont.@15',
    'outlier ρ': r'outlier $\rho$',
}

ADDPLOT_HDR_MAP = {
    'method': 'method',
    'anomaly distance ÷ bulk radius (med)': r'anom.\ dist $\div$ bulk radius (med)',
    'min': 'min',
    'source-cluster attribution acc.': 'attribution acc.',
    'angle to own cluster ° (med)': r'angle to own cluster $^\circ$ (med)',
    'same-pair angle ° (med)': r'pair angle $^\circ$ (med)',
    'bulk-control ratio (≈1 ideal)': r'control ratio ($\approx$1 ideal)',
}

SPECS = [
    ('density', '## density', 'tab:density', 'Non-uniform density'),
    ('clusters', '## clusters', 'tab:clusters', 'Distinct dense clusters'),
    ('transition', '## transition', 'tab:transition',
     'Continuous closed-loop transition'),
    ('outliers', '## outliers', 'tab:outliers', 'Off-subspace outliers'),
    ('populations', '## imbalanced populations', 'tab:populations',
     'Imbalanced two populations (95\\% vs.\\ 5\\%)'),
]


def get_table(md, after_heading, max_rows=7):
    idx = md.find(after_heading)
    assert idx >= 0, f"heading not found: {after_heading}"
    seg = md[idx:idx + 9000]
    lines = [l for l in seg.split('\n') if l.strip().startswith('|')]
    header = [c.strip() for c in lines[0].strip('|').split('|')]
    body = [[c.strip() for c in l.strip('|').split('|')]
            for l in lines[2:2 + max_rows]]
    return header, body


def texify(c):
    flag = '⚠' in c
    c = c.replace('⚠', '').strip()
    bold = ital = False
    m = re.match(r'^\*\*(.+)\*\*$', c)
    if m:
        bold, c = True, m.group(1)
    else:
        m = re.match(r'^\*(.+)\*$', c)
        if m:
            ital, c = True, m.group(1)
    c = c.strip()
    c = re.sub(r'\[([^\]]+)\]', r'{\\scriptsize[\1]}', c)
    if 'not operable' in c:
        c = re.sub(r'not operable: (.*)',
                   r'\\multicolumn{6}{l}{\\textit{not operable: \1}}', c)
    if bold and not flag:
        c = r'\textbf{' + c + '}'
    if ital:
        c = r'\textit{' + c + '}'
    if flag:
        c = c + r'\,\textsuperscript{\dag}'
    return c


def make_note(extra):
    return (r"Composite points: 1st$\to$5 \dots\ 5th$\to$1 on the full-$\rho$ "
            r"order and on the near-$\rho$ order" + extra + r"; rows sorted by "
            r"$\Sigma$. $\rho$ columns are Shepard (Spearman) correlations "
            r"vs.-ambient. \textbf{Bold} = best in column, \textit{italic} = "
            r"worst; \textsuperscript{\dag} = outright-failure flag (negative "
            r"near-band $\rho$, or worst tight-cluster crush exceeding "
            r"$5\times$). Brackets are bootstrap 95\% CIs over $R{=}3$ seeds; "
            r"deterministic methods (PCA, Isomap, toorPIA) show point values. "
            r"recall/trust/continuity are the variable-radius $k$-NN reference "
            r"block (biased; unscored). Values transcribed verbatim from the "
            r"v1.2.0 committed results.")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--report', default='REPORT.md')
    ap.add_argument('--outdir', default='paper/tables')
    args = ap.parse_args()

    md = open(args.report, encoding='utf-8').read()
    os.makedirs(args.outdir, exist_ok=True)

    for name, heading, label, title in SPECS:
        header, body = get_table(md, heading)
        ncol = len(header)
        hdr_tex = ' & '.join(COL_HDR_MAP.get(h, h) for h in header)
        rows_tex = [' & '.join(texify(c) for c in row) + r' \\'
                    for row in body]
        note = make_note(' and on the outlier-$\\rho$ order'
                         if name == 'outliers' else '')
        colspec = 'l' + 'c' * (ncol - 1)
        tex = ("% AUTO-GENERATED from REPORT.md --- do not hand-edit values\n"
               "\\begin{table}[t]\n\\centering\n"
               f"\\caption{{{title}: distance-fidelity ranking at "
               f"SNR${{=}}1$.\\; {note}}}\n"
               f"\\label{{{label}}}\n"
               "\\resizebox{\\textwidth}{!}{%\n"
               f"\\begin{{tabular}}{{{colspec}}}\n\\toprule\n"
               f"{hdr_tex} \\\\\n\\midrule\n"
               + '\n'.join(rows_tex)
               + "\n\\bottomrule\n\\end{tabular}}%\n\\end{table}\n")
        out = os.path.join(args.outdir, f'{name}.tex')
        open(out, 'w', encoding='utf-8').write(tex)
        print(f"wrote {out} ({ncol} cols)")

    header, body = get_table(md, '## Supplement — addplot')
    hdr_tex = ' & '.join(ADDPLOT_HDR_MAP.get(h, h) for h in header)
    rows_tex = []
    for row in body:
        cells = [texify(c) for c in row]
        if 'multicolumn' in cells[1]:
            rows_tex.append(cells[0] + ' & ' + cells[1] + r' \\')
        else:
            rows_tex.append(' & '.join(cells) + r' \\')
    tex = ("% AUTO-GENERATED from REPORT.md --- do not hand-edit values\n"
           "\\begin{table}[t]\n\\centering\n"
           "\\caption{Out-of-sample (addplot) monitoring test at SNR${=}1$: "
           "cluster-anchored anomalies (3\\,Rg along dimensions the normal "
           "basemap never varies in) and 50 fresh normal controls, added one "
           "at a time to a basemap fitted on normal data only. Detection $=$ "
           "anomaly distance from the map centroid over the bulk's median "
           "radius (large $=$ visibly outside); attribution $=$ direction "
           "from the centroid identifies the source cluster. PCA/Isomap use "
           "\\texttt{transform}; UMAP a seeded \\texttt{transform}; toorPIA "
           "server-side \\texttt{addplot\\_embedding}. t-SNE, PyMDE, and PCC "
           "expose no out-of-sample operation. Values transcribed verbatim "
           "from the v1.2.0 committed results.}\n"
           "\\label{tab:addplot}\n"
           "\\resizebox{\\textwidth}{!}{%\n"
           "\\begin{tabular}{lcccccc}\n\\toprule\n"
           + hdr_tex + " \\\\\n\\midrule\n"
           + '\n'.join(rows_tex)
           + "\n\\bottomrule\n\\end{tabular}}%\n\\end{table}\n")
    out = os.path.join(args.outdir, 'addplot.tex')
    open(out, 'w', encoding='utf-8').write(tex)
    print(f"wrote {out}")


if __name__ == '__main__':
    main()
