"""Calibration of the two SIE legs in rsi.tex.

Identification (Proposition in the Quantification section): along a historical path
with research inputs growing at rate g_R and a steady growth rate of the technology
level, the two legs of the SIE condition reduce to returns ratios

    r_E = g_nu / g_R                       (efficiency leg:  r_E > 1)
    r_C = k * g_h_sw / ((1 - sigma) g_R)   (capability leg:  r_C > 1)

where g_nu is the growth rate of fixed-task efficiency, g_h_sw the software-attributable
growth rate of the METR time horizon, k the Pareto tail index of the task-length
distribution, sigma the task elasticity of substitution and g_R the growth rate of
research labor. The binding returns r* = min{r_E, r_C}; r* - 1 is the asymptotic
elasticity of the growth rate with respect to cumulative research.

Running this script regenerates the table fragments (inputs, sources, Monte Carlo
results, Sobol sensitivity of the SIE indicator, tail fits, task-length summary, number
macros) and the figures used by rsi.tex: in drafts/tables/ and drafts/figures/ in the
paper repo, or in output/tables/ and output/figures/ in the standalone empirics repo. All
rates are natural logs per year.

    python calibration/calibrate.py
"""
from __future__ import annotations

import math
from statistics import NormalDist
from dataclasses import dataclass, field, replace
from pathlib import Path

import numpy as np

import matplotlib

import sys

sys.path.insert(0, str(Path(__file__).resolve().parent))
import tail_fit  # noqa: E402
import model_figure  # noqa: E402

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
from style import save_pdf  # noqa: E402  (also applies the paper figure style)

ROOT = Path(__file__).resolve().parents[1]
# Outputs go next to the LaTeX source in the paper repo, and to output/ in the
# standalone empirics repo (which has no drafts/ folder).
OUTPUT = ROOT / "drafts" if (ROOT / "drafts").is_dir() else ROOT / "output"
TABLES = OUTPUT / "tables"
FIGURES = OUTPUT / "figures"
TABLES.mkdir(parents=True, exist_ok=True)
FIGURES.mkdir(parents=True, exist_ok=True)

LN2 = math.log(2.0)


def rate_from_doubling_months(m: float) -> float:
    return LN2 / (m / 12.0)


def rate_from_doubling_days(d: float) -> float:
    return LN2 / (d / 365.0)


# --------------------------------------------------------------------------------------
# Inputs. Each carries a baseline, a range used one-at-a-time in the sensitivity analysis
# and as the support of the Monte Carlo distribution, and a source string for the paper.
# --------------------------------------------------------------------------------------


@dataclass
class Input:
    key: str
    symbol: str
    label: str
    baseline: float
    low: float
    high: float
    dist: str  # "lognormal" (baseline = median, low/high = 95% interval), "loguniform" or "uniform"
    source: str
    notes: str = ""
    fmt: str = "{:.2f}"


def lognormal_sd(low: float, high: float, level: float = 0.95) -> float:
    """Log standard deviation implied by a central interval at the given level."""
    return (math.log(high) - math.log(low)) / (2 * NormalDist().inv_cdf((1 + level) / 2))


# Gundlach et al. 2025 account for 6,930x of efficiency gains over 2012--2023, of which 2,700x is
# scale-dependent; the remaining ~2.6x is the scale-independent component.
G_NU_SCALEFREE = math.log(6930 / 2700) / (2023 - 2012)
G_NU_COMPONENTS = (
    rate_from_doubling_months(8.4),    # Ho et al. 2024, pretraining perplexity
    G_NU_SCALEFREE,                    # Gundlach et al. 2025, scale-independent component
    rate_from_doubling_months(9.0),    # Erdil--Besiroglu 2022, ImageNet
    math.log(3.221),                   # Gundlach et al. 2026, Table 2: GPQA Diamond, Pareto, open weight
)
# Each component's dispersion comes from its own source interval, in growth-rate space.
G_NU_SD = lognormal_sd(rate_from_doubling_months(14.3), rate_from_doubling_months(4.5))   # Ho et al., 95%
G_NU_SDS = (
    G_NU_SD,
    G_NU_SD,                                                                    # no interval reported
    lognormal_sd(rate_from_doubling_months(25.0), rate_from_doubling_months(4.0)),  # Erdil--Besiroglu, 95%
    lognormal_sd(math.log(1.536), math.log(6.754), 0.90),                       # Gundlach et al., 90%
)
# Inference is primary (it is the model's object); the other three only triangulate.
G_NU_WEIGHTS = (1 / 6, 1 / 6, 1 / 6, 0.50)
# Linear pool: the sources are noisy measures of one rate. The baseline is the weighted average of the
# four rates and the pooled log-sd is the weighted average of the sources' log-sds. A log-linear pool
# would weight each source by its precision, and the scale-independent precision is borrowed, not reported.
G_NU_POOLED = sum(w * g for w, g in zip(G_NU_WEIGHTS, G_NU_COMPONENTS))
G_NU_POOLED_SD = sum(w * sd for w, sd in zip(G_NU_WEIGHTS, G_NU_SDS))

INPUTS: dict[str, Input] = {}


def add(inp: Input) -> None:
    INPUTS[inp.key] = inp


add(Input(
    key="g_nu", symbol=r"$g_\nu$",
    label="Fixed-task efficiency growth",
    baseline=G_NU_POOLED,
    low=G_NU_POOLED * math.exp(-1.959964 * G_NU_POOLED_SD),
    high=G_NU_POOLED * math.exp(1.959964 * G_NU_POOLED_SD),
    dist="lognormal",
    source=r"Linear pool of four fixed-task rates: the baseline is their weighted average and the log standard deviation "
           r"is the weighted average of their log standard deviations, each from its source interval. Sources: the "
           r"hardware-adjusted algorithmic component of inference-price declines of \citet{gundlach2026price}, $3.22\times$/yr on "
           r"GPQA Diamond for open-weight models (90\% CI $1.54$--$6.75$) (weight 1/2); the scale-independent component of "
           r"\citet{gundlach2025origin}, $2.6\times$ over 2012--2023, no interval reported, pretraining dispersion (1/6); "
           r"\citet{ho2024algorithmic}, compute to fixed perplexity halves every 8.4 months (95\% CI 4.5--14.3) (1/6); the "
           r"ImageNet rate of \citet{erdil2022cv}, 9 months (95\% CI 4--25) (1/6)",
    notes="pretraining, 2012--2023",
))
add(Input(
    key="g_R", symbol=r"$g_R$",
    label="Research labor growth",
    baseline=0.86,                    # mean of ln(2.34) (OpenAI staff) and ln(2.4) (Epoch lab headcount)
    low=round(0.86 ** 2 / 1.15, 2),   # mirrors the high end in logs around the baseline
    high=1.15,
    dist="loguniform",
    source=r"Frontier-lab staff growth since late 2022: OpenAI staff $0.85$/yr \citep{howhitfill2025experiments}, Epoch AI lab headcount $\approx140\%$/yr \citep{davidson2026networks}; baseline is their mean in logs; "
           r"high end is the experiment-compute-augmented denominator of \citet{howhitfill2025experiments}, low end mirrors it in logs around the baseline",
))
add(Input(
    key="g_h", symbol=r"$g_h$",
    label="Time-horizon growth (50\\%)",
    baseline=rate_from_doubling_days(131.0),   # METR Time Horizon 1.1, since 2023
    low=rate_from_doubling_days(161.0),
    high=rate_from_doubling_days(107.0),
    dist="lognormal",
    source=r"\citet{metr2026th11}: doubling every 131 days since 2023 (95\% CI 107--161); the 2019--2025 doubling time is 196 days (162--223)",
    notes="2019--2025 trend: 196 days",
))
add(Input(
    key="s_sw", symbol=r"$s$",
    label="Software share of horizon growth",
    baseline=1.265 / (1.265 + math.log(4.5)),
    low=0.667 / (0.667 + math.log(4.5)),
    high=3.614 / (3.614 + math.log(4.5)),
    dist="uniform",
    source=r"\citet{whitfill2025horizon}: horizon growth proportional to effective-compute growth $g_C + g_{\mathrm{alg}}$, with algorithmic progress $g_{\mathrm{alg}} = 1.27$/yr (95\% CI 0.67--3.61) "
           r"and training compute growth $4.5\times$/yr \citep{sevilla2024compute}; the capability-index decomposition of \citet{ho2025rosetta} gives $0.52$",
))
add(Input(
    key="k", symbol=r"$k$",
    label="Task-length tail index",
    baseline=round(tail_fit.baseline_k(), 2),
    low=round(tail_fit.thin_k(), 2),
    high=round(tail_fit.rct_k(), 2),
    dist="loguniform",
    source=r"Length-weighted Pareto fit above four hours to the human-baselined tasks of METR's Time Horizon 1.1 suite "
           r"\citep{metr2026th11,rein2025hcast}; low end: the same fit above one hour; high end: the same fit to the "
           r"real open-source issues of \citet{becker2025productivity}; GDPval quantiles \citep{openai2025gdpval} give a count-weighted $1.34$",
))
add(Input(
    key="sigma", symbol=r"$\sigma$",
    label="Task substitution elasticity",
    baseline=0.61,
    low=0.0,
    high=0.83,
    dist="uniform",
    source=r"\citet{whitfillwu2025compute}: $-0.10$ (SE $0.18$) with frontier-experiment compute, $2.58$ (SE $0.34$) with total research compute; "
           r"\citet{davidson2023compute}: $0.67$; \citet{erdil2025gate}: $0.61$ ($0.17$--$0.83$); \citet{korinek2024scenarios}: $0.5$",
))

ORDER = ["g_nu", "g_R", "g_h", "s_sw", "k", "sigma"]

# Alternative point values referenced by scenarios (documented in the paper's text).
ALT = {
    "g_nu_scalefree": G_NU_SCALEFREE,                    # Gundlach et al. 2025 scale-independent component, ~2.6x over 2012--2023
    "g_nu_inference": math.log(3.221),                    # hardware-adjusted algorithmic inference-price decline, GPQA Diamond
    "g_nu_imagenet": rate_from_doubling_months(9.0),      # Erdil--Besiroglu 2022, ImageNet fixed-performance halving
    "g_h_longrun": rate_from_doubling_days(195.8),        # METR Time Horizon 1.1, 2019--2025 doubling
    "g_h_recent": rate_from_doubling_days(131.0),         # METR Time Horizon 1.1, post-2023 stitched doubling
    "g_h_2024": rate_from_doubling_days(89.0),            # METR Time Horizon 1.1, post-2024
    "g_R_sector": math.log(1.23),                         # sector research+engineering CAGR
    "g_R_augmented": 1.15,                                # 0.67*1.3 + 0.33*0.85
    "s_sw_rosetta": 0.52,                                 # Rosetta Stone: 0.297 software vs 0.27 compute capability units/yr
    "s_sw_pretraining": math.log(2.7) / (math.log(2.7) + math.log(4.5)),  # Ho et al. 2024 pretraining rate vs compute
    "sigma_ww_frontier": 0.0,                             # Whitfill--Wu frontier-experiment specification (point estimate -0.10, i.e. Leontief)
    "s_sw_low": 0.667 / (0.667 + math.log(4.5)),          # low end of the Whitfill--Snodin--Becker CI on algorithmic progress
    "k_thin": round(tail_fit.thin_k(), 2),                # TH1.1 baselined tasks, length-weighted fit above one hour
    "sigma_ww_total": 2.58,                               # Whitfill--Wu total-research-compute specification
}


# --------------------------------------------------------------------------------------
# Core formulas
# --------------------------------------------------------------------------------------


def legs(g_nu, g_R, g_h, s_sw, k, sigma):
    """Return (r_E, r_C). r_C is +inf when sigma >= 1 (capability leg vacuous)."""
    r_E = g_nu / g_R
    g_h_sw = s_sw * g_h
    with np.errstate(divide="ignore"):
        r_C = np.where(np.asarray(sigma) < 1.0, k * g_h_sw / ((1.0 - np.asarray(sigma)) * g_R), np.inf)
    return r_E, r_C


def normalized_parameters(g_nu, g_R, g_h, s_sw, k, sigma):
    """Elasticities under the normalization eta = 1 (A is the efficiency index)."""
    beta = g_R / g_nu
    mu = k * s_sw * g_h / g_nu
    return dict(eta=1.0, beta=beta, mu=mu, mu_over_1ms=mu / (1 - sigma) if sigma < 1 else math.inf)


# --------------------------------------------------------------------------------------
# Scenarios (point calibrations)
# --------------------------------------------------------------------------------------


@dataclass
class Scenario:
    name: str
    overrides: dict = field(default_factory=dict)
    note: str = ""

    def values(self) -> dict:
        v = {key: INPUTS[key].baseline for key in ORDER}
        v.update(self.overrides)
        return v


SCENARIOS = [Scenario("Baseline", {}, "all inputs at baseline")]


# --------------------------------------------------------------------------------------
# Monte Carlo
# --------------------------------------------------------------------------------------


def sample(inp: Input, n: int, rng: np.random.Generator) -> np.ndarray:
    if inp.dist == "lognormal":
        mu = math.log(inp.baseline)
        sd = (math.log(inp.high) - math.log(inp.low)) / (2 * 1.959964)
        return np.exp(rng.normal(mu, sd, n))
    if inp.dist == "loguniform":
        return np.exp(rng.uniform(math.log(inp.low), math.log(inp.high), n))
    if inp.dist == "uniform":
        return rng.uniform(inp.low, inp.high, n)
    raise ValueError(inp.dist)


def monte_carlo(n: int = 200_000, seed: int = 20260911, single_lognormal_g_nu: bool = False) -> dict:
    rng = np.random.default_rng(seed)
    inputs = dict(INPUTS)
    if single_lognormal_g_nu:  # comparison: one log-normal at the pretraining estimate
        inputs["g_nu"] = replace(INPUTS["g_nu"], baseline=rate_from_doubling_months(8.4),
                                 low=rate_from_doubling_months(14.3), high=rate_from_doubling_months(4.5))
    draws = {key: sample(inputs[key], n, rng) for key in ORDER}
    r_E, r_C = legs(**draws)
    r_star = np.minimum(r_E, r_C)
    both = (r_E > 1) & (r_C > 1)
    only_E = (r_E > 1) & ~(r_C > 1)
    only_C = ~(r_E > 1) & (r_C > 1)
    neither = ~(r_E > 1) & ~(r_C > 1)
    q = lambda x: np.quantile(x, [0.05, 0.25, 0.5, 0.75, 0.95])  # noqa: E731
    ci = lambda x: np.quantile(x, [0.025, 0.5, 0.975])  # 95% interval and median  # noqa: E731
    return dict(
        draws=draws, r_E=r_E, r_C=r_C, r_star=r_star,
        p_both=both.mean(), p_only_E=only_E.mean(), p_only_C=only_C.mean(), p_neither=neither.mean(),
        p_E=(r_E > 1).mean(), p_C=(r_C > 1).mean(),
        q_E=q(r_E), q_C=q(r_C), q_star=q(r_star),
        ci_E=ci(r_E), ci_C=ci(r_C), ci_star=ci(r_star),
    )


# --------------------------------------------------------------------------------------
# Table writers
# --------------------------------------------------------------------------------------


def fmt_r(x: float) -> str:
    if math.isinf(x):
        return r"$\infty$"
    return f"{x:.2f}"


def input_range(inp: Input) -> tuple[float, float]:
    return inp.low, inp.high


DIST_LABELS = {"loguniform": "log-uniform", "lognormal": "log-normal", "uniform": "uniform"}


def write_inputs_table() -> None:
    rows = []
    for key in ORDER:
        inp = INPUTS[key]
        lo, hi = input_range(inp)
        rng_txt = f"{inp.fmt.format(lo)}--{inp.fmt.format(hi)}"
        rows.append(f"{inp.symbol} & {inp.label} & {inp.fmt.format(inp.baseline)} & {rng_txt} & {DIST_LABELS[inp.dist]} \\\\")
    body = "\n".join(rows)
    (TABLES / "inputs.tex").write_text(
        "\\begin{tabular}{@{}llccl@{}}\n\\toprule\n"
        "Symbol & Input & Baseline & Range & Distribution \\\\\n\\midrule\n"
        f"{body}\n" + "\\bottomrule\n\\end{tabular}\n"
    )
    # Sources table (appendix)
    rows = []
    for key in ORDER:
        inp = INPUTS[key]
        rows.append(f"{inp.symbol} & {inp.source} \\\\")
    body = "\n".join(rows)
    (TABLES / "sources.tex").write_text(
        "\\begin{tabular}{@{}lp{0.86\\textwidth}@{}}\n\\toprule\n"
        "Input & Source \\\\\n\\midrule\n"
        f"{body}\n" + "\\bottomrule\n\\end{tabular}\n"
    )


def write_mc_table(mc: dict) -> None:
    """Headline table: baseline returns, their Monte Carlo medians, 95% intervals and shares above one,
    and the share of draws in each quadrant of the (r_E, r_C) plane."""
    base_rE, base_rC = legs(**{key: INPUTS[key].baseline for key in ORDER})
    pct = lambda x: f"{100 * x:.0f}\\%"  # noqa: E731

    def row(name, base, ci, share):
        return f"{name} & {base:.2f} & {ci[1]:.2f} & {ci[0]:.2f}--{ci[2]:.2f} & {pct(share)} \\\\"

    rows = [
        row("Efficiency returns $r_E$", float(base_rE), mc["ci_E"], mc["p_E"]),
        row("Capability returns $r_C$", float(base_rC), mc["ci_C"], mc["p_C"]),
        row("Binding returns $r^{*}=\\min\\{r_E,r_C\\}$", float(min(base_rE, base_rC)), mc["ci_star"], mc["p_both"]),
    ]
    outcomes = [
        ("both legs hold (an SIE)", mc["p_both"]),
        ("only the capability leg holds", mc["p_only_C"]),
        ("only the efficiency leg holds", mc["p_only_E"]),
        ("neither leg holds", mc["p_neither"]),
    ]
    (TABLES / "headline.tex").write_text(
        "\\begin{tabular}{@{}lcccc@{}}\n\\toprule\n"
        " & & \\multicolumn{3}{c}{Monte Carlo} \\\\\n\\cmidrule(lr){3-5}\n"
        " & Baseline & Median & 95\\% interval & Share above one \\\\\n\\midrule\n"
        + "\n".join(rows) + "\n\\midrule\n"
        "\\multicolumn{5}{@{}l}{\\emph{Share of Monte Carlo draws in which}} \\\\\n"
        + "\n".join(f"\\quad {name} & & & & {pct(x)} \\\\" for name, x in outcomes)
        + "\n\\bottomrule\n\\end{tabular}\n"
    )
    # Macros for in-text numbers
    base = {key: INPUTS[key].baseline for key in ORDER}
    r_E, r_C = legs(**base)
    macros = [
        f"\\newcommand{{\\rEbase}}{{{float(r_E):.2f}}}",
        f"\\newcommand{{\\rElo}}{{{mc['ci_E'][0]:.2f}}}",
        f"\\newcommand{{\\rEhi}}{{{mc['ci_E'][2]:.2f}}}",
        f"\\newcommand{{\\rEmedian}}{{{mc['ci_E'][1]:.2f}}}",
        f"\\newcommand{{\\rClo}}{{{mc['ci_C'][0]:.2f}}}",
        f"\\newcommand{{\\rChi}}{{{mc['ci_C'][2]:.2f}}}",
        f"\\newcommand{{\\rCmedian}}{{{mc['ci_C'][1]:.2f}}}",
        f"\\newcommand{{\\rstarlo}}{{{mc['ci_star'][0]:.2f}}}",
        f"\\newcommand{{\\rstarhi}}{{{mc['ci_star'][2]:.2f}}}",
        f"\\newcommand{{\\rEcapcut}}{{{100 * (1 - 1 / float(r_E)):.0f}\\%}}",  # cut in r_E that reaches the threshold
        f"\\newcommand{{\\pEbindspct}}{{{100 * np.mean(mc['r_E'] < mc['r_C']):.0f}\\%}}",  # efficiency is the binding leg
        f"\\newcommand{{\\rCrEratio}}{{{float(r_C) / float(r_E):.1f}}}",
        f"\\newcommand{{\\rEdoubling}}{{{2 ** float(r_E):.1f}}}",  # efficiency gain per doubling of research
        f"\\newcommand{{\\rCalphadoubling}}{{{2 ** (float(r_C) * (1 - base['sigma'])):.1f}}}",  # fall in the human-only share per doubling
        f"\\newcommand{{\\rCdoubling}}{{{2 ** float(r_C):.0f}}}",  # capability-ceiling gain per doubling
        f"\\newcommand{{\\pbothcutpct}}{{{100 * np.mean((mc['r_E'] / float(r_E) > 1) & (mc['r_C'] > 1)):.0f}\\%}}",  # both legs after that cut
        f"\\newcommand{{\\rCbase}}{{{float(r_C):.2f}}}",
        f"\\newcommand{{\\gnubase}}{{{base['g_nu']:.2f}}}",
        f"\\newcommand{{\\gRbase}}{{{base['g_R']:.2f}}}",
        f"\\newcommand{{\\ghbase}}{{{base['g_h']:.2f}}}",
        f"\\newcommand{{\\ghswbase}}{{{base['g_h'] * base['s_sw']:.2f}}}",
        f"\\newcommand{{\\sigmabase}}{{{base['sigma']:.2f}}}",
        f"\\newcommand{{\\sigmahigh}}{{{INPUTS['sigma'].high:.2f}}}",
        f"\\newcommand{{\\pboth}}{{{mc['p_both']:.2f}}}",
        f"\\newcommand{{\\pC}}{{{mc['p_C']:.2f}}}",
        f"\\newcommand{{\\ponlyE}}{{{mc['p_only_E']:.2f}}}",
        f"\\newcommand{{\\ponlyC}}{{{mc['p_only_C']:.2f}}}",
        f"\\newcommand{{\\pneither}}{{{mc['p_neither']:.2f}}}",
        f"\\newcommand{{\\pbothpct}}{{{100 * mc['p_both']:.0f}\\%}}",
        f"\\newcommand{{\\pCpct}}{{{100 * mc['p_C']:.0f}\\%}}",
        f"\\newcommand{{\\pEpct}}{{{100 * mc['p_E']:.0f}\\%}}",
        f"\\newcommand{{\\pEsinglepct}}{{{100 * monte_carlo(single_lognormal_g_nu=True)['p_E']:.0f}\\%}}",
        f"\\newcommand{{\\ponlyEpct}}{{{100 * mc['p_only_E']:.0f}\\%}}",
        f"\\newcommand{{\\ponlyCpct}}{{{100 * mc['p_only_C']:.0f}\\%}}",
        f"\\newcommand{{\\pneitherpct}}{{{100 * mc['p_neither']:.0f}\\%}}",
        f"\\newcommand{{\\rstarmedian}}{{{mc['q_star'][2]:.2f}}}",
        f"\\newcommand{{\\gnuscalefree}}{{{ALT['g_nu_scalefree']:.2f}}}",
        f"\\newcommand{{\\gnuimagenet}}{{{ALT['g_nu_imagenet']:.2f}}}",
        f"\\newcommand{{\\gnupretraining}}{{{rate_from_doubling_months(8.4):.2f}}}",
        f"\\newcommand{{\\gnuinference}}{{{ALT['g_nu_inference']:.2f}}}",
        f"\\newcommand{{\\gnusd}}{{{G_NU_POOLED_SD:.2f}}}",
        f"\\newcommand{{\\gRlow}}{{{INPUTS['g_R'].low:.2f}}}",
        f"\\newcommand{{\\gRlowfactor}}{{{math.exp(INPUTS['g_R'].low):.2f}}}",
        f"\\newcommand{{\\rEpooledlow}}{{{math.log(6.0) / base['g_R']:.1f}}}",
        f"\\newcommand{{\\rEpooledhigh}}{{{math.log(10.0) / base['g_R']:.1f}}}",
        f"\\newcommand{{\\ghlongrun}}{{{ALT['g_h_longrun']:.2f}}}",
        f"\\newcommand{{\\sswbase}}{{{base['s_sw']:.2f}}}",
        f"\\newcommand{{\\sswlow}}{{{INPUTS['s_sw'].low:.2f}}}",
        f"\\newcommand{{\\sswhigh}}{{{INPUTS['s_sw'].high:.2f}}}",
        f"\\newcommand{{\\kdoublingpct}}{{{100 * (1 - 2 ** -base['k']):.0f}\\%}}",  # automated by one horizon doubling
        f"\\newcommand{{\\kbase}}{{{base['k']:.2f}}}",
        f"\\newcommand{{\\kbaseloglog}}{{{tail_fit.baseline_k_loglog():.2f}}}",
        f"\\newcommand{{\\kthin}}{{{ALT['k_thin']:.2f}}}",
        f"\\newcommand{{\\krct}}{{{INPUTS['k'].high:.2f}}}",
        f"\\newcommand{{\\kgdpval}}{{{tail_fit.gdpval_k():.2f}}}",
    ]
    (TABLES / "macros.tex").write_text("\n".join(macros) + "\n")


SENSITIVITY_LABELS = {
    "g_nu": "Efficiency progress $g_\\nu$",
    "g_R": "Research input growth $g_R$",
    "k": "Tail index $k$",
    "sigma": "Substitution elasticity $\\sigma$",
    "s_sw": "Software share $s$",
    "g_h": "Time-horizon growth $g_h$",
}


def sie_sensitivity(n: int = 200_000, seed: int = 20261023) -> dict:
    """Variance-based sensitivity of the SIE indicator 1{r_E > 1, r_C > 1} to each input.

    First-order Sobol index S_i = Var(E[Y | x_i]) / Var(Y): the expected share of the variance of
    the indicator removed by learning input i alone. Estimated with the pick-freeze design of
    Saltelli et al. (2010): two independent n-draw samples A and B, and for each input the sample
    A_B^i that takes input i from B and every other input from A, giving
    S_i = mean(f(B) (f(A_B^i) - f(A))) / Var(Y). The share removed by learning every input other
    than g_nu jointly is 1 - S_T(g_nu), with the total-order index estimated as
    S_T(g_nu) = mean((f(A) - f(A_B^g_nu))^2) / (2 Var(Y)) (Jansen's estimator).
    """
    rng = np.random.default_rng(seed)
    A = {key: sample(INPUTS[key], n, rng) for key in ORDER}
    B = {key: sample(INPUTS[key], n, rng) for key in ORDER}

    def sie(d: dict) -> np.ndarray:
        r_E, r_C = legs(**d)
        return ((r_E > 1) & (r_C > 1)).astype(float)

    fA, fB = sie(A), sie(B)
    var = np.var(np.concatenate([fA, fB]))
    first, total = {}, {}
    for key in ORDER:
        fAB = sie({**A, key: B[key]})
        first[key] = float(np.mean(fB * (fAB - fA)) / var)
        total[key] = float(0.5 * np.mean((fA - fAB) ** 2) / var)
    others = 1 - total["g_nu"]
    capability = sum(first[k] for k in ("g_h", "s_sw", "k", "sigma"))

    rows = [f"{SENSITIVITY_LABELS[key]} & {first[key]:.2f} \\\\" for key in sorted(ORDER, key=lambda k: -first[k])]
    rows.append(f"\\midrule\nAll inputs except $g_\\nu$, jointly & {others:.2f} \\\\")
    (TABLES / "sie_sensitivity.tex").write_text(
        "\\begin{tabular}{@{}lc@{}}\n\\toprule\n"
        "Input & Share of SIE uncertainty \\\\\n\\midrule\n"
        + "\n".join(rows) + "\n\\bottomrule\n\\end{tabular}\n"
    )
    with (TABLES / "macros.tex").open("a") as f:
        f.write(f"\\newcommand{{\\sieshareGnu}}{{{100 * first['g_nu']:.0f}\\%}}\n")
        f.write(f"\\newcommand{{\\sieshareOthers}}{{{100 * others:.0f}\\%}}\n")
        f.write(f"\\newcommand{{\\sieshareGR}}{{{100 * first['g_R']:.0f}\\%}}\n")
        f.write(f"\\newcommand{{\\sieshareCap}}{{{100 * capability:.0f}\\%}}\n")
        f.write(f"\\newcommand{{\\sieSampleSize}}{{{n:,}}}\n".replace(",", "{,}"))
    print("SIE sensitivity, first-order:", {k: round(v, 3) for k, v in first.items()},
          "| others jointly:", round(others, 3))
    return {"first": first, "total": total, "others": others}


def sie_by_efficiency(mc: dict, groups: int = 5) -> list[tuple[float, float, float]]:
    """Share of Monte Carlo draws that are an SIE within each fifth of the efficiency-progress draws."""
    sie = (mc["r_E"] > 1) & (mc["r_C"] > 1)
    g = mc["draws"]["g_nu"]
    edges = np.quantile(g, np.linspace(0, 1, groups + 1))
    rows = []
    for lo, hi in zip(edges[:-1], edges[1:]):
        inside = (g >= lo) & (g <= hi)
        rows.append((float(lo), float(hi), float(sie[inside].mean())))
    with (TABLES / "macros.tex").open("a") as f:
        f.write(f"\\newcommand{{\\sieGnuLowpct}}{{{100 * rows[0][2]:.0f}\\%}}\n")
        f.write(f"\\newcommand{{\\sieGnuHighpct}}{{{100 * rows[-1][2]:.0f}\\%}}\n")
    return rows


# --------------------------------------------------------------------------------------
# Figures
# --------------------------------------------------------------------------------------


def plot_plane(mc: dict, scen_results) -> None:
    import ktviz
    from matplotlib.ticker import FixedLocator, FuncFormatter, NullLocator
    from matplotlib.transforms import blended_transform_factory

    fig, ax = ktviz.figure((6.4, 5.6))
    r_E, r_C = mc["r_E"], mc["r_C"]
    finite = np.isfinite(r_C)
    ax.scatter(r_E[finite][:20000], r_C[finite][:20000], s=3, alpha=0.3, color=ktviz.CONTEXT, lw=0,
               rasterized=True, zorder=2)
    # Centre the plot on the thresholds (1, 1): one log-symmetric extent for both axes, the smallest
    # round value that contains the 99th percentile of |ln r| on either axis.
    reach = max(np.quantile(np.abs(np.log(r_E[finite])), 0.99), np.quantile(np.abs(np.log(r_C[finite])), 0.99))
    ext = next(e for e in (2, 3, 5, 10, 15, 20, 30, 50, 100, 200, 500, 1000) if math.log(e) >= reach)
    xlim = ylim = (1 / ext, ext)
    ax.fill_between([1, xlim[1]], 1, ylim[1], color=ktviz.ACCENT, alpha=0.08, lw=0, zorder=1)
    ax.axvline(1, color=ktviz.INK, lw=0.8, zorder=3)
    ax.axhline(1, color=ktviz.INK, lw=0.8, zorder=3)
    lx, ly = ext ** 0.62, ext ** 0.88   # quadrant labels, placed symmetrically near the corners
    quad = dict(ha="center", va="center", fontsize=ktviz.LABEL_SIZE, zorder=4)
    ax.text(lx, ly, f"SIE: {mc['p_both']:.0%}", color=ktviz.ACCENT, fontweight=600, **quad)
    ax.text(1 / lx, ly, f"Capability only: {mc['p_only_C']:.0%}", color=ktviz.MUTED, **quad)
    ax.text(lx, 1 / ly, f"Efficiency only: {mc['p_only_E']:.0%}", color=ktviz.MUTED, **quad)
    ax.text(1 / lx, 1 / ly, f"Neither: {mc['p_neither']:.0%}", color=ktviz.MUTED, **quad)
    _, rE0, rC0 = next(t for t in scen_results if t[0].name == "Baseline")
    ax.scatter([rE0], [rC0], s=55, color=ktviz.ACCENT, edgecolor=ktviz.BG, linewidth=1.2, zorder=5)
    ax.annotate("Baseline", xy=(rE0, rC0), xytext=(6, 6), textcoords="offset points", fontsize=ktviz.LABEL_SIZE,
                color=ktviz.ACCENT, fontweight=600, ha="left", va="bottom", zorder=6)
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlim(*xlim)
    ax.set_ylim(*ylim)
    # Marginal 95% intervals, with a dot at the median, just outside the top and right edges.
    bar = dict(color=ktviz.INK, lw=1.3, clip_on=False, solid_capstyle="butt")
    small = dict(fontsize=ktviz.SOURCE_SIZE, color=ktviz.MUTED)
    top = blended_transform_factory(ax.transData, ax.transAxes)
    lo, med, hi = mc["ci_E"]
    ax.plot([lo, hi], [1.04, 1.04], transform=top, **bar)
    for x in (lo, hi):
        ax.plot([x, x], [1.025, 1.055], transform=top, **bar)
    ax.plot([med], [1.04], "o", ms=4, color=ktviz.INK, transform=top, clip_on=False)
    ax.text(hi * 1.15, 1.04, f"95% interval of $r_E$: {lo:.2f}–{hi:.2f}", transform=top, ha="left", va="center", **small)
    right = blended_transform_factory(ax.transAxes, ax.transData)
    lo, med, hi = mc["ci_C"]
    ax.plot([1.04, 1.04], [lo, hi], transform=right, **bar)
    for y in (lo, hi):
        ax.plot([1.025, 1.055], [y, y], transform=right, **bar)
    ax.plot([1.04], [med], "o", ms=4, color=ktviz.INK, transform=right, clip_on=False)
    ax.text(1.08, med, f"95% interval\nof $r_C$:\n{lo:.2f}–{hi:.2f}", transform=right, ha="left", va="center", **small)
    ax.set_xlabel("Efficiency returns $r_E$")
    ax.set_ylabel("Capability returns $r_C$")
    fmt = FuncFormatter(lambda v, _: f"{v:g}")
    ticks = sorted({1.0} | {t for v in (2, 10, 50, 100, 500, 1000) if v <= ext for t in (v, 1 / v)})
    for axis in (ax.xaxis, ax.yaxis):
        axis.set_major_locator(FixedLocator(ticks))
        axis.set_minor_locator(NullLocator())
        axis.set_major_formatter(fmt)
    save_pdf(fig, FIGURES / "legs_plane")
    plt.close(fig)


# --------------------------------------------------------------------------------------


def main() -> None:
    tail_fit.write_table()
    tail_fit.summary_table()
    tail_fit.plot_survival()
    write_inputs_table()
    scen_results = []
    for sc in SCENARIOS:
        r_E, r_C = legs(**sc.values())
        scen_results.append((sc, float(r_E), float(r_C)))
    mc = monte_carlo()
    write_mc_table(mc)
    sie_sensitivity()
    sie_by_efficiency(mc)
    plot_plane(mc, scen_results)
    model_figure.plot_output_map(FIGURES / "output_map", sigma=INPUTS["sigma"].baseline)

    print("Baseline inputs:", {k: round(INPUTS[k].baseline, 3) for k in ORDER})
    for sc, r_E, r_C in scen_results:
        print(f"{sc.name:24s} r_E={r_E:5.2f} r_C={fmt_r(r_C):>6s} r*={fmt_r(min(r_E, r_C)):>6s}")
    print("Monte Carlo: P(both)=%.3f P(E)=%.3f P(C)=%.3f P(E only)=%.3f P(C only)=%.3f P(neither)=%.3f"
          % (mc["p_both"], mc["p_E"], mc["p_C"], mc["p_only_E"], mc["p_only_C"], mc["p_neither"]))
    print("Quantiles r_E:", np.round(mc["q_E"], 2), " r_C:", np.round(mc["q_C"], 2), " r*:", np.round(mc["q_star"], 2))
    base = {key: INPUTS[key].baseline for key in ORDER}
    print("Normalized parameters (eta=1):", {k: round(v, 3) for k, v in normalized_parameters(**base).items()})


if __name__ == "__main__":
    main()
