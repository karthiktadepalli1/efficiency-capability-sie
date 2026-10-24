# efficiency-capability-sie

Reproduction code for the empirical analysis in *"Efficiency vs Capability in
the Intelligence Explosion"* (Karthik Tadepalli, 2026).

## Reproducing

Requires Python 3.12 or later.

```sh
python3 -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
python calibration/calibrate.py
```

With [uv](https://docs.astral.sh/uv/) instead:

```sh
uv venv && uv pip install -r requirements.txt && .venv/bin/python calibration/calibrate.py
```

This regenerates everything in `output/`. All random draws are seeded, so a
rerun reproduces the tables exactly. Figures use the
[ktviz](https://github.com/karthiktadepalli1/ktviz) chart style, installed from
GitHub at a pinned tag.

## What's here

- `calibration/calibrate.py`: the main entry point. Declares every input at the
  top of the file (baseline, uncertainty range, distribution and source),
  computes both returns ratios, runs the Monte Carlo and the sensitivity
  analysis, and writes all tables and figures.
- `calibration/tail_fit.py`: fits the Pareto tail index of the task-length
  distribution from METR's task data.
- `calibration/model_figure.py`: Figure 1, research capacity across efficiency
  and capability (Proposition 1), which illustrates the model rather than the data.
- `calibration/style.py`: the figure style shared by the paper's figures.
- `calibration/data/`: the task-length data extracts, with a
  [README](calibration/data/README.md) giving the exact source of each file.
- `output/tables/`, `output/figures/`: everything the script generates.

## Correspondence with the paper

| Output | Paper |
|---|---|
| `output/figures/output_map.pdf` | Figure 1: research capacity across efficiency and capability |
| `output/tables/inputs.tex` | Table 2: calibration inputs |
| `output/tables/headline.tex` | Table 3: returns ratios and Monte Carlo results |
| `output/figures/legs_plane.pdf` | Figure 2: Monte Carlo draws of the two returns ratios |
| `output/tables/sie_sensitivity.tex` | Table 5: first-order Sobol indices of the SIE indicator |
| `output/tables/task_length_summary.tex` | Table 6: task-length data |
| `output/figures/tail_survival.pdf` | Figure 3: task-length survival and the fitted Pareto tail |
| `output/tables/tail_fits.tex` | Table 7: tail-index estimates |
| `output/tables/sources.tex` | Table 8: sources for the calibration inputs |
| `output/tables/macros.tex` | Every number cited in the text |

Table 1 (the model environment) and Table 4 (published estimates of the returns
to software research) are written by hand in the paper, so neither is generated
here.

## Data and license

The code is released under the MIT license (see `LICENSE`). The data in
`calibration/data/` are extracts of public METR releases; see
`calibration/data/README.md` for their sources, and consult those sources'
terms before reusing the data outside this reproduction. All other inputs are
published estimates, entered with their citations in `calibration/calibrate.py`.

## Citation

```
Tadepalli, K. (2026). Efficiency vs Capability in the Intelligence Explosion.
```
