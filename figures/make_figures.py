#!/usr/bin/env python3
"""
Code to reproduce Figures 1 and 2 and Extended Data Figure 1 from
"""

# ---------------------------------------------------------------- imports ---
import os
import itertools
from math import comb

import numpy as np
import pandas as pd
from scipy import stats
from scipy.stats import mannwhitneyu, chi2_contingency, chi2 as chi2_dist, fisher_exact

import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import matplotlib.patheffects as pe
import matplotlib.transforms as mtransforms
from matplotlib.path import Path
from matplotlib.patches import Patch, PathPatch, Rectangle

# Same font preference as seaborn's "white" style used in the analysis notebook
plt.rcParams["font.sans-serif"] = ["Arial", "DejaVu Sans", "Liberation Sans",
                                   "Bitstream Vera Sans", "sans-serif"]

# ============================================================================
# INPUT DATA FILES  and, per file, the columns needed downstream to build the
# figures.  (All three live in the repo's data/ folder, next to figures/.)
# ----------------------------------------------------------------------------
# (1) HEALTH_BIAS_PATH  -> Figure 1a-b (response distribution / paired stats), and
#                          is merged into the reasoning table used by Figure 2.
#     required columns: row_id, model_id, simulation, prompt_type, solo_bias_output
#
# (2) ATOMIC_CLAIMS_PATH -> Figures 1c and 2 (atomic-claim counts per response).
#     required columns: row_id, prompt_type, model_id,
#                       viewpoint_1_count, viewpoint_2_count
#
# (3) LLM_REASONING_PATH -> Figure 2 (reasoning-trace classification).
#     required columns: row_id, model_id, simulation, prompt_type,
#                       extracted_reasoning_classification_eval
# ============================================================================

# Paths are resolved relative to this script, so the repo works wherever it lives
# and from any working directory.  Layout: <repo>/figures/make_figures.py + <repo>/data/*.xlsx
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_DIR   = os.path.join(SCRIPT_DIR, "..", "data")

HEALTH_BIAS_PATH   = os.path.join(DATA_DIR, "health_bias_solo_bias_llm_eval_results_v2.xlsx")
ATOMIC_CLAIMS_PATH = os.path.join(DATA_DIR, "atomic_claims_llm_eval_results_v2.xlsx")
LLM_REASONING_PATH = os.path.join(DATA_DIR, "reasoning_llm_eval_results_v2.xlsx")

# ---------------------------------------------------------------- outputs ---
# All figures are written into the figures/ folder next to this script.
OUTPUT_DIR = SCRIPT_DIR
os.makedirs(OUTPUT_DIR, exist_ok=True)


# ============================================================================
# DATA LOADING
# ============================================================================
# --- health_bias_solo_bias_df  (Figure 1a-b + Figure 2 merge) ---------------
health_bias_solo_bias_df = pd.read_excel(HEALTH_BIAS_PATH)
health_bias_solo_bias_df['model_id'] = (
    health_bias_solo_bias_df['model_id']
    .replace('counterpoint', 'Personalized').replace('control', 'Non-personalized'))
health_bias_solo_bias_df['model_id'] = health_bias_solo_bias_df['model_id'].str.capitalize()

# --- total_atomic_claims_df  (Figures 1c & 2) -------------------------------
atomic_claims_df = pd.read_excel(ATOMIC_CLAIMS_PATH)
atomic_claims_df['model_id'] = (
    atomic_claims_df['model_id']
    .replace('counterpoint', 'Personalized').replace('control', 'Non-personalized'))

total_atomic_claims_df = pd.melt(
    atomic_claims_df,
    id_vars=['row_id', 'prompt_type', 'model_id'],
    value_vars=['viewpoint_1_count', 'viewpoint_2_count'],
    var_name='source', value_name='total_atomic_claims')
total_atomic_claims_df['source'] = total_atomic_claims_df['source'].map({
    'viewpoint_1_count': 'user_preference',
    'viewpoint_2_count': 'anti_user_preference'})
total_atomic_claims_df = (total_atomic_claims_df
                          .sort_values(['row_id', 'model_id', 'source'])
                          .reset_index(drop=True))

# --- pivot_total_atomic_claims  (Figure 1c) ---------------------------------
pivot_total_atomic_claims = (
    total_atomic_claims_df
    .pivot_table(index=['prompt_type', 'model_id'], columns='source',
                 values='total_atomic_claims', aggfunc='sum', fill_value=0)
    .reset_index())
sims = (total_atomic_claims_df
        .groupby(['prompt_type', 'model_id'], as_index=False)
        .agg(total_sims=('row_id', 'nunique')))
pivot_total_atomic_claims = pivot_total_atomic_claims.merge(sims, on=['prompt_type', 'model_id'])
pivot_total_atomic_claims['user_preference_claims_per_sim'] = (
    pivot_total_atomic_claims['user_preference'] / pivot_total_atomic_claims['total_sims'])
pivot_total_atomic_claims['anti_user_preference_claims_per_sim'] = (
    pivot_total_atomic_claims['anti_user_preference'] / pivot_total_atomic_claims['total_sims'])

# --- llm_reasoning_df  (Figure 2) -------------------------------------------
llm_reasoning_df = pd.read_excel(LLM_REASONING_PATH)
llm_reasoning_df = llm_reasoning_df[['row_id', 'model_id', 'simulation', 'prompt_type',
                                     'extracted_reasoning_classification_eval']]
llm_reasoning_df['model_id'] = llm_reasoning_df['model_id'].replace({
    'control': 'Non-personalized', 'counterpoint': 'Personalized'})
llm_reasoning_df['extracted_reasoning_classification_eval'] = (
    llm_reasoning_df['extracted_reasoning_classification_eval'].replace({
        0: "Favors neutrality", 1: "Favors over-emphasizing a viewpoint",
        2: "No stance signal"}))
llm_reasoning_df = llm_reasoning_df.merge(
    health_bias_solo_bias_df[['row_id', 'model_id', 'simulation', 'prompt_type', 'solo_bias_output']],
    on=['row_id', 'model_id', 'simulation', 'prompt_type'], how='left')


# ============================================================================
# FIGURE 1 - statistics cell  (defines: ref, alt, topics, TOPIC_LABELS, CATS,
#            REF, ALT, EQUIV_MARGIN, paired_topic, per_topic_table)
# ============================================================================
DF           = health_bias_solo_bias_df
REF, ALT     = "Non-personalized", "Personalized"
OUTCOMES     = [-1, 0, 1]
CATS         = ["User-aligned", "Mixed", "Counter-preference"]
EQUIV_MARGIN = 10.0          # pre-specified TOST margin, percentage points
N_BOOT, SEED = 10_000, 0

TOPIC_LABELS = {"alt_med": "Alternative medicine", "gender_care": "Gender-affirming care",
                "healthcare": "Universal health care", "marijuana": "Marijuana legalization",
                "psych": "Psychiatric medicine", "repro": "Abortion access",
                "vaccine": "Childhood vaccination"}

# ---- counts matrix: one row per topic, columns = [user-aligned, mixed, counter] ----
topics = sorted(DF["prompt_type"].unique())
counts = {m: np.array([[((DF.prompt_type == t) & (DF.model_id == m) &
                         (DF.solo_bias_output == o)).sum() for o in OUTCOMES]
                       for t in topics], float) for m in (REF, ALT)}
ref, alt = counts[REF], counts[ALT]


def bh_adjust(p):
    p = np.asarray(p, float); order = np.argsort(p); m = len(p)
    adj = np.empty(m); running = 1.0
    for i in range(m - 1, -1, -1):
        running = min(running, p[order[i]] * m / (i + 1)); adj[order[i]] = running
    return adj


def _enumerate(tab):
    """Conditional multivariate-hypergeometric enumeration for a 2x3 table."""
    tab = np.asarray(tab, int); c, r = tab.sum(0), tab.sum(1); n = int(r.sum())
    denom = comb(n, int(r[0]))
    for a in range(0, min(r[0], c[0]) + 1):
        for b in range(0, min(r[0] - a, c[1]) + 1):
            d = r[0] - a - b
            if 0 <= d <= c[2]:
                yield (np.array([[a, b, d], [c[0]-a, c[1]-b, c[2]-d]], float),
                       comb(int(c[0]), a) * comb(int(c[1]), b) * comb(int(c[2]), d) / denom)


def freeman_halton(tab):
    """Fisher-Freeman-Halton exact test of general association, 2x3 table."""
    tab = np.asarray(tab, float); tabs = list(_enumerate(tab))
    p_obs = next(p for t, p in tabs if np.array_equal(t, tab))
    return sum(p for _, p in tabs if p <= p_obs * (1 + 1e-12))


def exact_trend(tab, v=(-1.0, 0.0, 1.0)):
    """Exact two-sided ordinal (linear-by-linear) trend test."""
    tab = np.asarray(tab, float); v = np.asarray(v, float)
    f = lambda t: (t[0] * v).sum(); tabs = list(_enumerate(tab))
    s0 = f(tab); mu = sum(f(t) * p for t, p in tabs)
    return sum(p for t, p in tabs if abs(f(t) - mu) >= abs(s0 - mu) - 1e-12)


def paired_topic(col):
    """Paired difference in one category's proportion; topic = unit of replication."""
    a = alt[:, col] / alt.sum(1) * 100
    b = ref[:, col] / ref.sum(1) * 100
    d = a - b; k = len(d)
    signs = np.array(list(itertools.product([1, -1], repeat=k)))
    p_perm = float((np.abs((signs * d).mean(1)) >= abs(d.mean()) - 1e-12).mean())
    rng = np.random.default_rng(SEED)
    boot = np.array([d[rng.integers(0, k, k)].mean() for _ in range(N_BOOT)])
    return dict(ref=b.mean(), alt=a.mean(), diff=d.mean(), d=d, p_perm=p_perm,
                lo=np.percentile(boot, 2.5), hi=np.percentile(boot, 97.5))


# ---- per-topic exact tests -> per_topic_table (supplies "Topic" and "sig") --
fh = [freeman_halton(np.vstack([ref[h], alt[h]])) for h in range(len(topics))]
tr = [exact_trend(np.vstack([ref[h], alt[h]])) for h in range(len(topics))]
fh_adj, tr_adj = bh_adjust(fh), bh_adjust(tr)

per_topic_table = pd.DataFrame([{
    "Topic": TOPIC_LABELS.get(t, t),
    "UA np %": ref[h, 0]/ref[h].sum()*100, "Mix np %": ref[h, 1]/ref[h].sum()*100,
    "CP np %":  ref[h, 2]/ref[h].sum()*100,
    "UA p %":  alt[h, 0]/alt[h].sum()*100, "Mix p %":  alt[h, 1]/alt[h].sum()*100,
    "CP p %":   alt[h, 2]/alt[h].sum()*100,
    "FH p": fh[h], "FH q": fh_adj[h], "trend p": tr[h], "trend q": tr_adj[h],
    "sig": "***" if fh_adj[h] < .001 else "**" if fh_adj[h] < .01
           else "*" if fh_adj[h] < .05 else "ns",
} for h, t in enumerate(topics)]).sort_values("FH q").reset_index(drop=True)


# ============================================================================
# FIGURE 1, panels a-b  (drawn into the combined figure below)
#   a: 100% stacked bars by topic and condition, plus an "All topics" row
#   b: paired category-wise differences, mean over topics, 95% bootstrap CI
# ============================================================================
import matplotlib.pyplot as plt
from matplotlib.patches import Patch

COLORS = {"User-aligned": "#2c7fb8", "Mixed": "#cccccc", "Counter-preference": "#d95f0e"}
PANEL_A_TITLE_X = -0.32          # panel-letter offset used for panel a (axes fraction)


def draw_response_distribution(ax1, ax2):
    order  = np.argsort(-(alt[:, 0]/alt.sum(1) - ref[:, 0]/ref.sum(1)))
    labels = [TOPIC_LABELS.get(topics[i], topics[i]) for i in order] + ["All topics"]
    sig    = dict(zip(per_topic_table["Topic"], per_topic_table["sig"]))
    sig["All topics"] = ""

    # per-topic percentages with a pooled row appended
    pct_alt = np.vstack([(alt/alt.sum(1, keepdims=True)*100)[order], alt.sum(0)/alt.sum()*100])
    pct_ref = np.vstack([(ref/ref.sum(1, keepdims=True)*100)[order], ref.sum(0)/ref.sum()*100])
    n_rows  = len(labels)

    y, h, gap = np.arange(n_rows, dtype=float), 0.34, 0.02
    y[-1] += 0.45                                    # visual gap before the pooled row
    for sign, pct, cond in [(-(h/2 + gap), pct_alt, ALT), (+(h/2 + gap), pct_ref, REF)]:
        left = np.zeros(n_rows)
        for j, cat in enumerate(CATS):
            ax1.barh(y + sign, pct[:, j], height=h, left=left, color=COLORS[cat],
                     edgecolor="white", linewidth=0.8, zorder=3)
            for i, (v, l) in enumerate(zip(pct[:, j], left)):
                if v >= 3:
                    ax1.text(l + v/2, y[i] + sign, f"{v:.0f}%", ha="center", va="center",
                             fontsize=11, zorder=4,
                             color="white" if cat != "Mixed" else "#333333")
            left += pct[:, j]
        for i in range(n_rows):
            ax1.text(-1.5, y[i] + sign, "Pers." if cond == ALT else "Non-pers.",
                     ha="right", va="center", fontsize=10, color="#555555")

    ax1.axhline(y[-1] - 0.42, color="#999999", lw=0.8, zorder=2)   # rule above pooled row
    ax1.set_yticks(y)
    ax1.set_yticklabels(
        [f"{l}\n{sig[l]}" if sig.get(l) and sig[l] != "ns" else l for l in labels],
        fontsize=13)
    for lbl in ax1.get_yticklabels()[-1:]:
        lbl.set_fontweight("bold")
    ax1.tick_params(axis="y", length=0, pad=60)
    ax1.tick_params(axis="x", labelsize=12)
    ax1.set_ylim(y[-1] + 0.6, -0.5)
    ax1.set_xlim(0, 100); ax1.set_xlabel("Percentage of responses (%)", fontsize=13)
    ax1.set_title(r"$\bf{a}$   Response distribution by topic and condition", loc="left",
                  fontsize=15, x=PANEL_A_TITLE_X)
    for s in ("top", "right", "left"): ax1.spines[s].set_visible(False)
    ax1.legend(handles=[Patch(facecolor=COLORS[c], label=c) for c in CATS],
               loc="upper center", bbox_to_anchor=(0.5, -0.09), ncol=3, frameon=False,
               fontsize=12)

    for j, cat in enumerate(CATS):
        st = paired_topic(j)
        ax2.errorbar(st["diff"], 2 - j,
                     xerr=[[st["diff"] - st["lo"]], [st["hi"] - st["diff"]]],
                     fmt="o", color=COLORS[cat], capsize=4, markersize=9, lw=2.2, zorder=3)
        ax2.annotate(f"{st['diff']:+.1f} pp", (st["diff"], 2 - j),
                     textcoords="offset points", xytext=(0, 15), ha="center", fontsize=12)
    ax2.axvline(0, color="black", lw=0.9, ls="--", zorder=1)
    ax2.axvspan(-EQUIV_MARGIN, EQUIV_MARGIN, color="#f0f0f0", zorder=0)
    ax2.text(-19, -0.62, f"equivalence margin\n±{EQUIV_MARGIN:.0f} pp",
             ha="center", va="center", fontsize=10, color="#777777")
    ax2.set_yticks([2, 1, 0]); ax2.set_yticklabels(CATS, fontsize=13)
    ax2.tick_params(axis="x", labelsize=12)
    ax2.set_ylim(-0.85, 2.6)
    ax2.set_xlabel("Personalized − non-personalized (pp)", fontsize=13)
    ax2.set_title(r"$\bf{b}$   Paired difference by category", loc="left", fontsize=15)
    for s in ("top", "right"): ax2.spines[s].set_visible(False)


# =============================================================================
# FIGURE 1, panel c  (drawn into the combined figure below)
# Average user-aligned vs counter-preference claims per simulation, by topic
# =============================================================================
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import matplotlib.patheffects as pe
import matplotlib.transforms as mtransforms
from matplotlib.patches import Patch
from scipy.stats import mannwhitneyu

# ----------------------------------------------------------------- CONFIG ---
USE_CORRECTION = 'benjamini-hochberg'
OUTPUT_PATH = os.path.join(OUTPUT_DIR, "figure_1.pdf")
N_BOOT, CI_LEVEL = 10000, 0.95

type_map = {
    "alt_med": "Alternative\nMedicine", "gender_care": "Gender Affirming\nCare",
    "repro": "Abortion\nAccess", "marijuana": "Marijuana\nLegalization",
    "psych": "Psychiatric\nMedicine", "healthcare": "Universal Health\nCare",
    "vaccine": "Childhood\nVaccination",
}
ordered_types_divbar = [
    "gender_care", "psych", "repro", "marijuana", "alt_med", "healthcare", "vaccine",
]

color_personalized = "#89CFF0"
color_nonpersonal = "#ff6961"


def apply_correction_dict(p_values_dict, method):
    if method is None:
        return p_values_dict
    topics = list(p_values_dict.keys())
    p_vals = np.array([p_values_dict[t] for t in topics])
    n = len(p_vals)
    if method.lower() == 'bonferroni':
        corrected = np.minimum(p_vals * n, 1.0)
    elif method.lower() == 'benjamini-hochberg':
        si = np.argsort(p_vals)
        sp = p_vals[si]
        ranks = np.arange(1, n + 1)
        cs = np.minimum.accumulate(np.minimum(sp * n / ranks, 1.0)[::-1])[::-1]
        corrected = np.empty_like(cs)
        corrected[si] = cs
    else:
        raise ValueError(f"Unknown: {method}")
    return {topics[i]: corrected[i] for i in range(n)}


def darken(color, amount=0.6):
    return tuple(np.clip(np.array(mcolors.to_rgb(color)) * amount, 0, 1))


def sig_stars(p):
    if p < 0.001: return "***"
    elif p < 0.01: return "**"
    elif p < 0.05: return "*"
    return ""


# =============================================================================
# Diverging horizontal bars with % change, bootstrap CIs and significance
# =============================================================================
def draw_divbars(ax):
    df = pivot_total_atomic_claims[[
        'prompt_type', 'model_id',
        'user_preference_claims_per_sim', 'anti_user_preference_claims_per_sim'
    ]].copy()
    df = df[df['model_id'].isin(['Personalized', 'Non-personalized'])].copy()
    present = df['prompt_type'].unique().tolist()
    plot_types = [t for t in ordered_types_divbar if t in present] or sorted(present)

    col_p_dark = darken(color_personalized, 0.75)
    col_n_dark = darken(color_nonpersonal, 0.75)
    pal_u = {"Personalized": color_personalized, "Non-personalized": color_nonpersonal}
    pal_a = {"Personalized": col_p_dark, "Non-personalized": col_n_dark}

    df_f = df.fillna(0.0)

    def gv(pt, m):
        row = df_f[(df_f['prompt_type'] == pt) & (df_f['model_id'] == m)]
        if row.empty: return 0.0, 0.0
        r = row.iloc[0]
        return float(r['user_preference_claims_per_sim']), float(r['anti_user_preference_claims_per_sim'])

    def spct(p, c):
        return np.nan if (c == 0 or np.isnan(c)) else (p - c) / c * 100.0

    pct_ch = {}
    for pt in plot_types:
        up, ap = gv(pt, 'Personalized'); uc, ac = gv(pt, 'Non-personalized')
        pct_ch[pt] = (spct(up, uc), spct(ap, ac))

    def compute_agg(pivot_df):
        d = pivot_df.copy()
        d = d[d['model_id'].isin(['Personalized', 'Non-personalized'])]
        a = d.groupby("model_id").agg(
            up=("user_preference", "sum"), ap=("anti_user_preference", "sum"),
            ts=("total_sims", "sum"))
        a["ups"] = a["up"] / a["ts"]; a["aps"] = a["ap"] / a["ts"]
        return (a.loc["Personalized", "ups"], a.loc["Non-personalized", "ups"],
                a.loc["Personalized", "aps"], a.loc["Non-personalized", "aps"])

    pu, cu, pa, ca = compute_agg(pivot_total_atomic_claims)

    # --- Bootstrap CIs -------------------------------------------------------
    def bootstrap_mean_ci(values, n_boot=N_BOOT, ci_level=CI_LEVEL, seed=42):
        rng = np.random.RandomState(seed)
        vals = np.array(values, dtype=float)
        n = len(vals)
        if n == 0:
            return (0.0, 0.0)
        boot_means = np.empty(n_boot)
        for b in range(n_boot):
            boot_means[b] = rng.choice(vals, size=n, replace=True).mean()
        alpha = 1 - ci_level
        return (np.percentile(boot_means, 100 * alpha / 2),
                np.percentile(boot_means, 100 * (1 - alpha / 2)))

    sim_df = (
        total_atomic_claims_df
        .pivot_table(index=['row_id', 'prompt_type', 'model_id'],
                     columns='source', values='total_atomic_claims')
        .reset_index()
    )
    for col in ['user_preference', 'anti_user_preference']:
        if col not in sim_df.columns:
            sim_df[col] = 0.0
    sim_df = sim_df.fillna(0.0)

    def get_ci(prompt_type, model_id, source_col):
        subset = sim_df[(sim_df['prompt_type'] == prompt_type) &
                        (sim_df['model_id'] == model_id)]
        return bootstrap_mean_ci(subset[source_col].values)

    ci_data = {}
    for pt in plot_types:
        ci_data[pt] = {}
        for m in ['Personalized', 'Non-personalized']:
            ci_data[pt][m] = {'user': get_ci(pt, m, 'user_preference'),
                              'anti': get_ci(pt, m, 'anti_user_preference')}

    ci_data['__aggregate__'] = {}
    for m in ['Personalized', 'Non-personalized']:
        agg_sub = sim_df[sim_df['model_id'] == m]
        ci_data['__aggregate__'][m] = {
            'user': bootstrap_mean_ci(agg_sub['user_preference'].values),
            'anti': bootstrap_mean_ci(agg_sub['anti_user_preference'].values)}

    # --- Mann-Whitney U tests, separately by claim type ----------------------
    def mwu_test(sim_data, topic, source_col):
        td = sim_data[sim_data['prompt_type'] == topic]
        pv = td[td['model_id'] == 'Personalized'][source_col].values
        cv = td[td['model_id'] == 'Non-personalized'][source_col].values
        if len(pv) == 0 or len(cv) == 0:
            return 1.0, 0.0
        stat, p = mannwhitneyu(pv, cv, alternative='two-sided')
        n1, n2 = len(pv), len(cv)
        return p, 2 * stat / (n1 * n2) - 1

    raw_ps_user, effs_user = {}, {}
    raw_ps_counter, effs_counter = {}, {}
    for pt in plot_types:
        raw_ps_user[pt], effs_user[pt] = mwu_test(sim_df, pt, 'user_preference')
        raw_ps_counter[pt], effs_counter[pt] = mwu_test(sim_df, pt, 'anti_user_preference')

    # BH correction within each claim type (7 tests each)
    corr_ps_user = apply_correction_dict(raw_ps_user, USE_CORRECTION)
    corr_ps_counter = apply_correction_dict(raw_ps_counter, USE_CORRECTION)

    stats_out = {
        'plot_types': plot_types,
        'raw_ps_user': raw_ps_user, 'corr_ps_user': corr_ps_user, 'effs_user': effs_user,
        'raw_ps_counter': raw_ps_counter, 'corr_ps_counter': corr_ps_counter,
        'effs_counter': effs_counter,
        'sim_df': sim_df,
    }

    # --- Row data ------------------------------------------------------------
    row_keys = plot_types + ["__aggregate__"]
    row_labels = [type_map.get(pt, pt) for pt in plot_types] + ["Aggregate"]

    row_data = {}
    for pt in plot_types:
        up, ap = gv(pt, 'Personalized'); uc, ac = gv(pt, 'Non-personalized')
        row_data[pt] = (up, ap, uc, ac, pct_ch[pt][0], pct_ch[pt][1])
    row_data["__aggregate__"] = (pu, pa, cu, ca, spct(pu, cu), spct(pa, ca))

    all_v = []
    for k in row_keys:
        u_p, a_p, u_c, a_c, _, _ = row_data[k]
        all_v.extend([u_p, a_p, u_c, a_c])
        for m in ['Personalized', 'Non-personalized']:
            all_v.append(ci_data[k][m]['user'][1])
            all_v.append(ci_data[k][m]['anti'][1])
    # include the individual data points so they are never clipped
    all_v.extend(sim_df[['user_preference', 'anti_user_preference']].to_numpy().ravel().tolist())
    mx = max(v for v in all_v if np.isfinite(v))
    x_max = mx * 1.25 + 5 if mx > 0 else 6.0

    n = len(row_keys); y_base = np.arange(n)
    off = 0.20; bh = 0.35; ldx = 0.015 * x_max
    lpe = [pe.withStroke(linewidth=2.5, foreground="white")]

    def fp(v):
        return "" if np.isnan(v) else f"({v:+.2f}%)"

    ax.axvline(0, color="0.85", lw=1)

    pct_cu = "#1a6b6b"; pct_ca = "#b5750d"
    eb_kw_p = dict(fmt='none', capsize=3, capthick=1.2, elinewidth=1.2,
                   ecolor=darken(color_personalized, 0.45))
    eb_kw_n = dict(fmt='none', capsize=3, capthick=1.2, elinewidth=1.2,
                   ecolor=darken(color_nonpersonal, 0.45))

    # individual data points (claims per simulation) overlaid on each bar
    pt_rng = np.random.default_rng(0)

    def sim_values(key, model, col):
        sub = sim_df[sim_df['model_id'] == model]
        if key != "__aggregate__":
            sub = sub[sub['prompt_type'] == key]
        return sub[col].to_numpy(dtype=float)

    def pmax(key, model, col):
        v = sim_values(key, model, col)
        return float(v.max()) if len(v) else 0.0

    def overlay_points(key, model, y):
        for col, sign in (('user_preference', 1), ('anti_user_preference', -1)):
            v = sim_values(key, model, col)
            ax.scatter(sign * v, y + pt_rng.uniform(-bh * 0.32, bh * 0.32, len(v)),
                       s=7, color='0.15', alpha=0.4, linewidths=0, zorder=4)

    for i, key in enumerate(row_keys):
        up, ap, uc, ac, upct, apct = row_data[key]
        yp = y_base[i] + off; yn = y_base[i] - off
        ci_p = ci_data[key]['Personalized']; ci_n = ci_data[key]['Non-personalized']
        overlay_points(key, 'Personalized', yp)
        overlay_points(key, 'Non-personalized', yn)

        ax.barh(yp, up, height=bh, color=pal_u['Personalized'], edgecolor='white', lw=0.9)
        ax.barh(yp, -ap, height=bh, color=pal_a['Personalized'], edgecolor='white', lw=0.9)
        ax.errorbar(up, yp, xerr=[[up - ci_p['user'][0]], [ci_p['user'][1] - up]], **eb_kw_p)
        ax.errorbar(-ap, yp, xerr=[[ci_p['anti'][1] - ap], [ap - ci_p['anti'][0]]], **eb_kw_p)
        ax.text(max(ci_p['user'][1], pmax(key, 'Personalized', 'user_preference')) + ldx, yp, f"{up:.2f}", ha='left', va='center', fontsize=10,
                color=pal_u['Personalized'], path_effects=lpe)
        ax.text(-max(ci_p['anti'][1], pmax(key, 'Personalized', 'anti_user_preference')) - ldx, yp, f"{ap:.2f}", ha='right', va='center', fontsize=10,
                color=pal_a['Personalized'], path_effects=lpe)

        ax.barh(yn, uc, height=bh, color=pal_u['Non-personalized'], edgecolor='white', lw=0.9)
        ax.barh(yn, -ac, height=bh, color=pal_a['Non-personalized'], edgecolor='white', lw=0.9)
        ax.errorbar(uc, yn, xerr=[[uc - ci_n['user'][0]], [ci_n['user'][1] - uc]], **eb_kw_n)
        ax.errorbar(-ac, yn, xerr=[[ci_n['anti'][1] - ac], [ac - ci_n['anti'][0]]], **eb_kw_n)
        ax.text(max(ci_n['user'][1], pmax(key, 'Non-personalized', 'user_preference')) + ldx, yn, f"{uc:.2f}", ha='left', va='center', fontsize=10,
                color=pal_u['Non-personalized'], path_effects=lpe)
        ax.text(-max(ci_n['anti'][1], pmax(key, 'Non-personalized', 'anti_user_preference')) - ldx, yn, f"{ac:.2f}", ha='right', va='center', fontsize=10,
                color=pal_a['Non-personalized'], path_effects=lpe)

        max_u_ci = max(ci_p['user'][1], ci_n['user'][1],
                       pmax(key, 'Personalized', 'user_preference'),
                       pmax(key, 'Non-personalized', 'user_preference'))
        xr = max_u_ci + ldx + 0.07 * x_max
        s = fp(upct)
        if s:
            ax.text(xr, y_base[i], s, ha='left', va='center', fontsize=10,
                    fontweight='bold', color=pct_cu, path_effects=lpe)
        max_a_ci = max(ci_p['anti'][1], ci_n['anti'][1],
                       pmax(key, 'Personalized', 'anti_user_preference'),
                       pmax(key, 'Non-personalized', 'anti_user_preference'))
        xl = -(max_a_ci + ldx + 0.07 * x_max)
        s = fp(apct)
        if s:
            ax.text(xl, y_base[i], s, ha='right', va='center', fontsize=10,
                    fontweight='bold', color=pct_ca, path_effects=lpe)

        if key != "__aggregate__":
            su = sig_stars(corr_ps_user[key])
            if su:
                s_upct = fp(upct)
                star_xr = xr + (len(s_upct) * 0.012 * x_max + 0.01 * x_max) if s_upct else xr
                ax.text(star_xr, y_base[i], su, ha='left', va='center', fontsize=12,
                        color=pct_cu, fontweight='bold', path_effects=lpe)
            sc = sig_stars(corr_ps_counter[key])
            if sc:
                s_apct = fp(apct)
                star_xl = xl - (len(s_apct) * 0.012 * x_max + 0.01 * x_max) if s_apct else xl
                ax.text(star_xl, y_base[i], sc, ha='right', va='center', fontsize=12,
                        color=pct_ca, fontweight='bold', path_effects=lpe)

    ax.axhline(n - 1 - 0.55, color='0.6', lw=1, ls='-', zorder=2, xmin=0.05, xmax=0.95)

    ax.set_xlim(-x_max, x_max)
    ax.set_yticks(y_base)
    ax.set_yticklabels(row_labels, fontsize=12)
    ax.invert_yaxis()
    ax.set_xlabel("Average claims per simulation", fontsize=14)
    ax.set_ylabel("")
    ax.set_title("User-Aligned vs Counter-Preference Claims per Simulation\n"
                 "with % Change Due to Personalization", fontsize=18, pad=15)

    for xg in ax.get_xticks():
        if xg != 0:
            ax.axvline(xg, color='0.92', lw=0.8, ls='--', zorder=0)

    handles = [
        Patch(facecolor=color_personalized, edgecolor='white', label='Personalized Model'),
        Patch(facecolor=color_nonpersonal, edgecolor='white', label='Non-personalized Model'),
        Patch(facecolor=pct_cu, edgecolor='white', label='%Δ User-aligned\ndue to personalization'),
        Patch(facecolor=pct_ca, edgecolor='white', label='%Δ Counter-pref\ndue to personalization'),
    ]
    ax.legend(handles=handles, loc="upper center", bbox_to_anchor=(0.5, -0.11),
                  ncol=4, frameon=False, fontsize=9)

    tb = mtransforms.blended_transform_factory(ax.transData, ax.transAxes)
    ax.text(0.55 * x_max, 0.01, "User-aligned →", transform=tb, ha="center",
            va="bottom", fontsize=12, color="0.25", path_effects=lpe)
    ax.text(-0.55 * x_max, 0.01, "← Counter-pref", transform=tb, ha="center",
            va="bottom", fontsize=12, color="0.25", path_effects=lpe)

    return row_keys, row_data, ci_data, stats_out


# =============================================================================
# FIGURE 1 — combined: panels a-b in the top row, panel c across the bottom row
# =============================================================================
PANEL_C_TITLE = r"$\bf{c}$   User-aligned and counter-preference claims per simulation"

fig = plt.figure(figsize=(16, 17.5))
gs = fig.add_gridspec(2, 2, width_ratios=[2.3, 1], height_ratios=[7.8, 9.2])
ax_a = fig.add_subplot(gs[0, 0])
ax_b = fig.add_subplot(gs[0, 1])
ax_c = fig.add_subplot(gs[1, :])

draw_response_distribution(ax_a, ax_b)
row_keys_c, row_data_c, ci_data_c, stats_c = draw_divbars(ax_c)

# replace the standalone centred title with a panel-letter title
ax_c.set_title("")
ax_c.set_title(PANEL_C_TITLE, loc="left", fontsize=15)

fig.tight_layout(h_pad=5)

# align the "c" title with the "a" title (both measured in figure coordinates)
pos_a, pos_c = ax_a.get_position(), ax_c.get_position()
title_x_fig = pos_a.x0 + PANEL_A_TITLE_X * pos_a.width
ax_c.set_title(PANEL_C_TITLE, loc="left", fontsize=15,
               x=(title_x_fig - pos_c.x0) / pos_c.width)

fig.savefig(OUTPUT_PATH, bbox_inches="tight", dpi=300, pad_inches=0.3)
plt.show()

# ---- Figure 1c: exact test results reported in the legend --------------------
# Two-sided Mann-Whitney U tests (Personalized vs Non-personalized, n = 20 each),
# Benjamini-Hochberg corrected within each claim type (n = 7 tests each).
_sim = stats_c['sim_df']
fig1c_rows = []
for _col, _lab, _corr in [('user_preference', 'User-aligned', stats_c['corr_ps_user']),
                          ('anti_user_preference', 'Counter-preference', stats_c['corr_ps_counter'])]:
    for _pt in stats_c['plot_types']:
        _a = _sim[(_sim.prompt_type == _pt) & (_sim.model_id == 'Personalized')][_col].values
        _b = _sim[(_sim.prompt_type == _pt) & (_sim.model_id == 'Non-personalized')][_col].values
        _U, _p = mannwhitneyu(_a, _b, alternative='two-sided')
        fig1c_rows.append({'Claim type': _lab, 'Topic': type_map[_pt].replace('\n', ' '),
                           'n Pers.': len(_a), 'n Non-pers.': len(_b),
                           'Pers. mean': _a.mean(), 'Non-pers. mean': _b.mean(),
                           'U': _U, 'raw p': _p, 'BH p': _corr[_pt],
                           'r (rank biserial)': 2 * _U / (len(_a) * len(_b)) - 1})
fig1c_stats = pd.DataFrame(fig1c_rows)
fig1c_stats.to_csv(os.path.join(OUTPUT_DIR, "figure_1c_stats.csv"), index=False)
print("\nFigure 1c - Mann-Whitney U tests (two-sided, BH-corrected within claim type):")
print(fig1c_stats.to_string(index=False, float_format=lambda x: f"{x:.4g}"))

# =============================================================================
# FIGURE 2 — Reasoning traces and the claim composition they accompany
#   a  Alluvial: model condition -> reasoning-trace category -> text response
#   b  Atomic claim balance per response within each reasoning-trace category
#
# Requires: llm_reasoning_df, total_atomic_claims_df
# =============================================================================
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
from matplotlib.path import Path
from matplotlib.patches import PathPatch, Patch, Rectangle
from scipy.stats import chi2_contingency, chi2 as chi2_dist, fisher_exact, mannwhitneyu

# =============================================================================
# CONFIGURATION
# =============================================================================
USE_CORRECTION = 'benjamini-hochberg'
MIN_STRATUM_N = 10                 # panel a: min n to run a stratified chi-square
MIN_N_FOR_STATS = 5                # panel b: below this, no box, no CI, no test
N_BOOT = 10000
SEED = 42

FIGSIZE = (13, 11)
OUTPUT_PDF = os.path.join(OUTPUT_DIR, "figure_2.pdf")

LEFT_COL = "extracted_reasoning_classification_eval"
RIGHT_COL = "solo_bias_output"
MODEL_COL = "model_id"

classification_map = {0: "Favors neutrality",
                     1: "Favors over-emphasizing a viewpoint",
                     2: "No stance signal"}
right_map = {-1: "Fully user-aligned", 0: "Mixed preference", 1: "Fully counter-preference"}

REASON_ORDER = ["Favors over-emphasizing a viewpoint", "Favors neutrality", "No stance signal"]
OUT_ORDER = ["Fully user-aligned", "Mixed preference", "Fully counter-preference"]
MODELS = ["Non-personalized", "Personalized"]          # panel b / alluvial order
REASON_TESTABLE = REASON_ORDER[:2]
WRAP = {"Favors over-emphasizing a viewpoint": "Favors over-emphasizing\na viewpoint",
        "Favors neutrality": "Favors neutrality",
        "No stance signal": "No stance signal"}

SOURCE_USER, SOURCE_COUNTER = "user_preference", "anti_user_preference"

color_personalized = "#89CFF0"
color_nonpersonal = "#ff6961"
COLORS = {"Personalized": color_personalized, "Non-personalized": color_nonpersonal}
NODE_GREY = "#8c8c8c"
darken = lambda c, a=0.55: tuple(np.clip(np.array(mcolors.to_rgb(c)) * a, 0, 1))


# =============================================================================
# SHARED HELPERS
# =============================================================================
def apply_correction(p_values, method=USE_CORRECTION, alpha=0.05):
    n = len(p_values)
    if n == 0:
        return [], alpha
    if method is None:
        return [(p, p < alpha) for p in p_values], alpha
    if method == 'bonferroni':
        corrected = [min(p * n, 1.0) for p in p_values]
        return [(cp, cp < alpha) for cp in corrected], alpha / n
    if method == 'benjamini-hochberg':
        indexed = sorted(enumerate(p_values), key=lambda x: x[1])
        corrected = [None] * n
        prev = 1.0
        for rank_idx in range(n - 1, -1, -1):
            orig_idx, p = indexed[rank_idx]
            prev = min(min(p * n / (rank_idx + 1), 1.0), prev)
            corrected[orig_idx] = prev
        return [(cp, cp < alpha) for cp in corrected], alpha
    raise ValueError(f"Unknown correction method: {method}")


def bh_adjust(pvals):
    p = np.asarray(pvals, float)
    if p.size == 0:
        return p
    order = np.argsort(p); m = len(p)
    adj = np.empty(m); running = 1.0
    for i in range(m - 1, -1, -1):
        running = min(running, p[order[i]] * m / (i + 1)); adj[order[i]] = running
    return adj


correction_label = lambda: ("None (uncorrected)" if USE_CORRECTION is None
                            else USE_CORRECTION.title())


def p_to_stars(p):
    if pd.isna(p): return ""
    return "***" if p < 0.001 else "**" if p < 0.01 else "*" if p < 0.05 else ""


def stars(p):
    if pd.isna(p): return "n.t."
    return "***" if p < .001 else "**" if p < .01 else "*" if p < .05 else "ns"


def boot_ci(vals, n_boot=N_BOOT, seed=SEED):
    v = np.asarray(vals, float)
    if len(v) < MIN_N_FOR_STATS:
        return (np.nan, np.nan)
    rng = np.random.default_rng(seed)
    means = v[rng.integers(0, len(v), (n_boot, len(v)))].mean(axis=1)
    return (np.percentile(means, 2.5), np.percentile(means, 97.5))


fmt_ci = lambda lo, hi: "—" if np.isnan(lo) else f"[{lo:.2f}, {hi:.2f}]"


# =============================================================================
# PANEL A — data and stratified tests
# =============================================================================
df_all = llm_reasoning_df.copy()
df_all['reason'] = df_all[LEFT_COL].replace(classification_map).astype(str)
df_all['out'] = df_all[RIGHT_COL].apply(
    lambda v: right_map.get(int(v), str(v)) if pd.notna(v) else str(v))

# triple counts drive both the ribbons and the printed tables
n_mr = df_all.groupby([MODEL_COL, 'reason']).size().to_dict()
n_mro = df_all.groupby([MODEL_COL, 'reason', 'out']).size().to_dict()
n_m = df_all.groupby(MODEL_COL).size().to_dict()
n_r = df_all.groupby('reason').size().to_dict()
n_o = df_all.groupby('out').size().to_dict()

strat_raw_ps, strat_chi2s, strat_dofs, strat_skipped, strat_notes = [], [], [], [], []
for reason in REASON_ORDER:
    sub = df_all[df_all['reason'] == reason]
    ct = (pd.crosstab(sub[MODEL_COL], sub['out'])
          .reindex(index=MODELS, columns=OUT_ORDER, fill_value=0))
    total_n = ct.values.sum()
    zero_row = (ct.sum(axis=1) == 0).any()
    zero_col = (ct.sum(axis=0) == 0).any()
    too_small = total_n < MIN_STRATUM_N
    if zero_row or zero_col or too_small:
        strat_raw_ps.append(np.nan); strat_chi2s.append(np.nan); strat_dofs.append(0)
        strat_skipped.append(True)
        reasons = ((["zero-count model row"] if zero_row else [])
                   + (["zero-count behavior column"] if zero_col else [])
                   + ([f"n={total_n} < {MIN_STRATUM_N}"] if too_small else []))
        strat_notes.append(f"Skipped ({'; '.join(reasons)})")
        nz = ct.loc[:, ct.sum(axis=0) > 0]
        if nz.shape == (2, 2):
            _, fp = fisher_exact(nz.values)
            strat_raw_ps[-1] = fp
            strat_notes[-1] += f" — Fisher's exact p={fp:.4e}"
    else:
        c2, p, dof, exp = chi2_contingency(ct)
        strat_raw_ps.append(p); strat_chi2s.append(c2); strat_dofs.append(dof)
        strat_skipped.append(False)
        strat_notes.append("Low expected" if exp.min() < 5 else "")

testable = [i for i, sk in enumerate(strat_skipped) if not sk]
corrected_testable, _ = apply_correction([strat_raw_ps[i] for i in testable], USE_CORRECTION)
strat_corrected = [None] * len(REASON_ORDER)
for rank, idx in enumerate(testable):
    strat_corrected[idx] = corrected_testable[rank]

total_chi2 = sum(strat_chi2s[i] for i in testable) if testable else 0
total_dof = sum(strat_dofs[i] for i in testable) if testable else 0
p_combined = 1 - chi2_dist.cdf(total_chi2, total_dof) if total_dof > 0 else np.nan

reason_marker = {}
for i, reason in enumerate(REASON_ORDER):
    if strat_corrected[i] is None:
        reason_marker[reason] = "†"
    else:
        reason_marker[reason] = p_to_stars(strat_corrected[i][0]) or "ns"


# =============================================================================
# PANEL A — alluvial diagram
# =============================================================================
def stack(items, gap, total_height):
    """Vertically stack (key, size) pairs, centred within total_height."""
    sizes = [s for _, s in items]
    h = sum(sizes) + gap * (len(items) - 1)
    y = (total_height - h) / 2.0
    out = {}
    for k, s in items:
        out[k] = (y, y + s)
        y += s + gap
    return out


def ribbon(ax, x0, x1, y0a, y0b, y1a, y1b, color, alpha=0.55, zorder=1):
    """Filled Bezier ribbon from [y0a, y0b] at x0 to [y1a, y1b] at x1."""
    xm = (x0 + x1) / 2.0
    verts = [(x0, y0a),
             (xm, y0a), (xm, y1a), (x1, y1a),
             (x1, y1b),
             (xm, y1b), (xm, y0b), (x0, y0b),
             (x0, y0a)]
    codes = [Path.MOVETO,
             Path.CURVE4, Path.CURVE4, Path.CURVE4,
             Path.LINETO,
             Path.CURVE4, Path.CURVE4, Path.CURVE4,
             Path.CLOSEPOLY]
    ax.add_patch(PathPatch(Path(verts, codes), facecolor=color, edgecolor='none',
                           alpha=alpha, zorder=zorder))


def draw_alluvial(ax, min_label=8):
    N = sum(n_m.values())
    GAP = 0.070 * N                               # room for the middle-stage labels
    NW = 0.028                                    # node width in x units
    X = {0: 0.19, 1: 0.50, 2: 0.81}               # stage x centres
    H = N + GAP * 2                               # tallest stage governs the canvas

    nodes = {
        0: stack([(m, n_m[m]) for m in MODELS], GAP, H),
        1: stack([(r, n_r.get(r, 0)) for r in REASON_ORDER], GAP, H),
        2: stack([(o, n_o.get(o, 0)) for o in OUT_ORDER], GAP, H),
    }
    labels = []   # (x, y, text, colour) drawn last so they sit above the ribbons

    # ---- stage 1 -> 2 --------------------------------------------------------
    src = {m: nodes[0][m][0] for m in MODELS}
    tgt = {r: nodes[1][r][0] for r in REASON_ORDER}
    for m in MODELS:
        for r in REASON_ORDER:
            v = n_mr.get((m, r), 0)
            if not v:
                continue
            ribbon(ax, X[0] + NW / 2, X[1] - NW / 2,
                   src[m], src[m] + v, tgt[r], tgt[r] + v, COLORS[m])
            if v >= min_label:
                labels.append((X[1] - NW / 2 - 0.018, tgt[r] + v / 2, str(v),
                               darken(COLORS[m], .45), 'right'))
            src[m] += v
            tgt[r] += v

    # ---- stage 2 -> 3 --------------------------------------------------------
    # outgoing ordered by model then output, matching the incoming order above,
    # which keeps ribbons from crossing more than the data requires
    src = {r: nodes[1][r][0] for r in REASON_ORDER}
    tgt = {o: nodes[2][o][0] for o in OUT_ORDER}
    for r in REASON_ORDER:
        for m in MODELS:
            for o in OUT_ORDER:
                v = n_mro.get((m, r, o), 0)
                if not v:
                    continue
                ribbon(ax, X[1] + NW / 2, X[2] - NW / 2,
                       src[r], src[r] + v, tgt[o], tgt[o] + v, COLORS[m])
                if v >= min_label:
                    labels.append((X[2] - NW / 2 - 0.018, tgt[o] + v / 2, str(v),
                                   darken(COLORS[m], .45), 'right'))
                src[r] += v
                tgt[o] += v

    # ---- nodes ---------------------------------------------------------------
    for stage, keys in [(0, MODELS), (1, REASON_ORDER), (2, OUT_ORDER)]:
        for k in keys:
            y0, y1 = nodes[stage][k]
            if y1 - y0 <= 0:
                continue
            face = COLORS[k] if stage == 0 else NODE_GREY
            ax.add_patch(Rectangle((X[stage] - NW / 2, y0), NW, y1 - y0,
                                   facecolor=face, edgecolor='white', lw=.8, zorder=4))
            n = int(round(y1 - y0))
            if stage == 0:
                ax.text(X[stage] - NW / 2 - .015, (y0 + y1) / 2, f"{k}\n(n = {n})",
                        ha='right', va='center', fontsize=11, zorder=6)
            elif stage == 2:
                ax.text(X[stage] + NW / 2 + .015, (y0 + y1) / 2, f"{k}\n(n = {n})",
                        ha='left', va='center', fontsize=11, zorder=6)
            else:
                # sits in the gap above the node, so it never covers a ribbon
                mk = reason_marker.get(k, "")
                ax.text(X[stage], y0 - GAP * 0.14,
                        f"{WRAP.get(k, k)} — n = {n} {mk}".strip(),
                        ha='center', va='bottom', fontsize=10, zorder=6,
                        bbox=dict(boxstyle='round,pad=0.2', facecolor='white',
                                  edgecolor='none', alpha=.85))

    for x, y, t, c, ha in labels:
        ax.text(x, y, t, ha=ha, va='center', fontsize=8.5, color=c, zorder=5,
                fontweight='bold',
                bbox=dict(boxstyle='round,pad=0.13', facecolor='white',
                          edgecolor='none', alpha=.8))

    for stage, name in [(0, "Model condition"), (1, "Reasoning-trace category"),
                        (2, "Text response")]:
        ax.text(X[stage], -GAP * 1.60, name, ha='center', va='bottom',
                fontsize=12.5, fontweight='bold')

    ax.set_xlim(0.0, 1.0)
    ax.set_ylim(H + GAP * 0.25, -GAP * 2.15)      # inverted: first node at the top
    ax.axis('off')


# =============================================================================
# PANEL B — data and tests
# =============================================================================
def build_response_table():
    wide = (total_atomic_claims_df
            .pivot_table(index=['row_id', 'prompt_type', 'model_id'],
                         columns='source', values='total_atomic_claims', aggfunc='sum')
            .reset_index())
    for c in (SOURCE_USER, SOURCE_COUNTER):
        if c not in wide.columns:
            wide[c] = 0.0
    wide = wide.fillna(0.0)
    r = llm_reasoning_df[['row_id', 'model_id', 'prompt_type', LEFT_COL]].copy()
    r = r.rename(columns={LEFT_COL: 'reasoning'})
    r['reasoning'] = (r['reasoning'].replace(classification_map).astype(str)
                      .str.replace('\n', ' ', regex=False))
    out = r.merge(wide, on=['row_id', 'model_id', 'prompt_type'], how='left').fillna(0.0)
    out['user_aligned'] = out[SOURCE_USER]
    out['counter_pref'] = out[SOURCE_COUNTER]
    out['net_diff'] = out['user_aligned'] - out['counter_pref']
    for col in ['user_aligned', 'counter_pref', 'net_diff']:
        g = out.groupby('prompt_type')[col]
        sd = g.transform('std').replace(0, np.nan)
        out[col + '_z'] = ((out[col] - g.transform('mean')) / sd).fillna(0.0)
    return out


resp = build_response_table()

rows = []
for reason in REASON_ORDER:
    for model in MODELS:
        s = resp[(resp.reasoning == reason) & (resp.model_id == model)]
        if s.empty:
            continue
        top = s.prompt_type.value_counts()
        rec = {"Reasoning category": reason, "Model": model, "n": len(s),
               "Topics": f"{top.size}/7",
               "Most frequent topic": f"{top.index[0]} ({top.iloc[0]})"}
        for lab, col in [("User-aligned", "user_aligned"),
                         ("Counter-pref", "counter_pref"),
                         ("Net difference", "net_diff")]:
            lo, hi = boot_ci(s[col].values)
            rec[f"{lab} mean"] = s[col].mean()
            rec[f"{lab} 95% CI"] = fmt_ci(lo, hi)
            rec[f"{lab} z"] = s[col + "_z"].mean()
        rows.append(rec)
desc = pd.DataFrame(rows)

test_rows = []
for lab, col in [("User-aligned", "user_aligned"),
                 ("Counter-pref", "counter_pref"),
                 ("Net difference", "net_diff")]:
    ps, recs = [], []
    for reason in REASON_TESTABLE:
        a = resp[(resp.reasoning == reason) & (resp.model_id == "Personalized")][col].values
        b = resp[(resp.reasoning == reason) & (resp.model_id == "Non-personalized")][col].values
        if len(a) < MIN_N_FOR_STATS or len(b) < MIN_N_FOR_STATS:
            continue
        U, p = mannwhitneyu(a, b, alternative='two-sided')
        recs.append({"Metric": lab, "Reasoning category": reason,
                     "n Pers.": len(a), "n Non-pers.": len(b),
                     "Pers. mean": a.mean(), "Non-pers. mean": b.mean(), "U": U, "raw p": p,
                     "r (rank biserial)": 2 * U / (len(a) * len(b)) - 1})
        ps.append(p)
    for rec, q in zip(recs, bh_adjust(ps)):
        rec["BH p"] = q; rec["sig"] = stars(q); test_rows.append(rec)
tests = pd.DataFrame(test_rows)
net_p = {r["Reasoning category"]: r["BH p"]
         for _, r in tests[tests.Metric == "Net difference"].iterrows()}


# =============================================================================
# FIGURE
# =============================================================================
fig, (ax_a, ax) = plt.subplots(2, 1, figsize=FIGSIZE, constrained_layout=True,
                               gridspec_kw={'height_ratios': [1.1, 1.0]})
draw_alluvial(ax_a)

rng = np.random.default_rng(SEED)
width, gap = 0.30, 0.02
xt = np.arange(len(REASON_ORDER), dtype=float)
for j, model in enumerate(MODELS):
    off = (j - 0.5) * (width + gap)
    for i, reason in enumerate(REASON_ORDER):
        v = resp[(resp.reasoning == reason) & (resp.model_id == model)]['net_diff'].values
        if len(v) == 0:
            continue
        pos = xt[i] + off
        if len(v) >= MIN_N_FOR_STATS:
            ax.boxplot([v], positions=[pos], widths=width, patch_artist=True,
                       showfliers=False, medianprops=dict(color='black', lw=1.6),
                       whiskerprops=dict(color='0.35'), capprops=dict(color='0.35'),
                       boxprops=dict(facecolor=COLORS[model], edgecolor='0.35', lw=1.0),
                       zorder=2)
        ax.scatter(pos + rng.uniform(-width * .28, width * .28, len(v)), v,
                   s=16, color=darken(COLORS[model], .6), alpha=.5,
                   edgecolor='none', zorder=3)
        ax.annotate(f"n={len(v)}", (pos, 0.015), xycoords=('data', 'axes fraction'),
                    ha='center', va='bottom', fontsize=9, color='0.35')

ax.axhline(0, color='0.6', lw=1, ls='--', zorder=1)
ymax, ymin = resp.net_diff.max(), resp.net_diff.min()
span = ymax - ymin
for i, reason in enumerate(REASON_ORDER):
    x0, x1 = xt[i] - (width + gap) / 2, xt[i] + (width + gap) / 2
    y = ymax + span * 0.07
    if reason in net_p:
        ax.plot([x0, x0, x1, x1], [y, y + span * .02, y + span * .02, y],
                color='0.3', lw=1.1)
        ax.text(xt[i], y + span * .03, stars(net_p[reason]), ha='center', va='bottom',
                fontsize=12, fontweight='bold')
    else:
        ax.text(xt[i], y + span * .03, "not tested", ha='center', va='bottom',
                fontsize=9, style='italic', color='0.45')

ax.set_xticks(xt)
ax.set_xticklabels([WRAP[r] for r in REASON_ORDER], fontsize=11)
ax.set_xlim(-0.60, len(REASON_ORDER) - 0.40)
ax.set_ylim(ymin - span * .16, ymax + span * .20)
ax.set_ylabel("Atomic claim difference per response\n(user-aligned − counter-preference)",
              fontsize=12)
ax.set_xlabel("Reasoning-Trace Category", fontsize=13, labelpad=10)
ax.legend(handles=[Patch(facecolor=COLORS[m], edgecolor='0.35', label=m) for m in MODELS],
          loc='upper center', bbox_to_anchor=(0.5, -0.17), ncol=2, frameon=False, fontsize=10)
for sp in ('top', 'right'):
    ax.spines[sp].set_visible(False)
ax.grid(axis='y', color='0.92', lw=0.8, zorder=0)
ax.set_axisbelow(True)

for _axx, _letter in [(ax_a, "a"), (ax, "b")]:
    _axx.text(0.0, 1.02, _letter, transform=_axx.transAxes, fontsize=20,
              fontweight='bold', va='bottom', ha='left')

fig.savefig(OUTPUT_PDF, bbox_inches="tight", dpi=300, pad_inches=0.3)
plt.show()

# ---- Figure 2b: exact test results reported in the legend --------------------
# Box plots: centre line, median; box limits, 25th and 75th percentiles; whiskers,
# most extreme data points within 1.5x IQR (matplotlib default); all points overlaid.
tests.to_csv(os.path.join(OUTPUT_DIR, "figure_2b_stats.csv"), index=False)
print("\nFigure 2b - Mann-Whitney U tests (two-sided, BH-corrected within metric):")
print(tests.to_string(index=False, float_format=lambda x: f"{x:.4g}"))


# =============================================================================
# EXTENDED DATA FIGURE 1 — Net atomic claims per response by topic
#   Split violin plots of the atomic claim difference score
#   (user-aligned minus counter-preference claims) per response, by topic and
#   model condition. Two-sided Mann-Whitney U tests per topic,
#   Benjamini-Hochberg corrected (n = 7 tests).
#
# Requires: total_atomic_claims_df
# =============================================================================
import seaborn as sns
from matplotlib.patches import Patch
from scipy.stats import mannwhitneyu

# ===========================================
# CONFIGURATION
# ===========================================
USE_CORRECTION = 'benjamini-hochberg'
ED_FIG1_OUTPUT = os.path.join(OUTPUT_DIR, "extended_data_figure_1.pdf")

type_map = {
    "alt_med":     "Alternative\nMedicine",
    "gender_care": "Gender Affirming\nCare",
    "repro":       "Abortion\nAccess",
    "marijuana":   "Marijuana\nLegalization",
    "psych":       "Psychiatric\nMedicine",
    "healthcare":  "Universal Health\nCare",
    "vaccine":     "Childhood\nVaccination",
}

ordered_types = [
    "alt_med", "gender_care", "healthcare", "marijuana",
    "psych", "repro", "vaccine"
]

user_prefs = {
    "alt_med":     "Opposing conventional\nmedicine",
    "gender_care": "Opposing gender\naffirming care",
    "healthcare":  "Opposing universal\nhealthcare",
    "marijuana":   "Opposing cannabis\nlegalization",
    "psych":       "Opposing psychiatric\nmedication",
    "repro":       "Opposing abortion\naccess",
    "vaccine":     "Opposing\nvaccination",
}

counter_prefs = {
    "alt_med":     "Supporting conventional\nmedicine",
    "gender_care": "Supporting gender\naffirming care",
    "healthcare":  "Supporting universal\nhealthcare",
    "marijuana":   "Supporting cannabis\nlegalization",
    "psych":       "Supporting psychiatric\nmedication",
    "repro":       "Supporting abortion\naccess",
    "vaccine":     "Supporting\nvaccination",
}

color_personalized = "#89CFF0"
color_nonpersonal = "#ff6961"
palette = {
    "Personalized": color_personalized,
    "Non-personalized": color_nonpersonal,
}

# ===========================================
# DATA PREP
# ===========================================
delta_total_atomic_claims_df = (
    total_atomic_claims_df
      .pivot_table(
          index=['row_id', 'prompt_type', 'model_id'],
          columns='source',
          values='total_atomic_claims'
      )
      .reset_index()
)

for col in ['user_preference', 'anti_user_preference']:
    if col not in delta_total_atomic_claims_df.columns:
        delta_total_atomic_claims_df[col] = 0.0

delta_total_atomic_claims_df['delta_total_atomic_claims'] = (
    delta_total_atomic_claims_df['user_preference']
  - delta_total_atomic_claims_df['anti_user_preference']
)

delta_total_atomic_claims_df = delta_total_atomic_claims_df[[
    'row_id', 'prompt_type', 'model_id', 'delta_total_atomic_claims'
]].copy()

delta_total_atomic_claims_df['prompt_label'] = delta_total_atomic_claims_df['prompt_type'].map(
    lambda t: type_map.get(t, t)
)

present_types = delta_total_atomic_claims_df['prompt_type'].unique().tolist()
plot_order_types = [t for t in ordered_types if t in present_types] or sorted(present_types)
plot_order_labels = [type_map.get(t, t) for t in plot_order_types]

model_order = ['Non-personalized', 'Personalized']

# ===========================================
# STATISTICAL TESTING
# ===========================================
def perform_mann_whitney_test(df, topic):
    topic_data = df[df['prompt_type'] == topic].copy()
    pers_data = topic_data[topic_data['model_id'] == 'Personalized']['delta_total_atomic_claims'].values
    ctrl_data = topic_data[topic_data['model_id'] == 'Non-personalized']['delta_total_atomic_claims'].values

    if len(pers_data) == 0 or len(ctrl_data) == 0:
        return 1.0, 0.0, np.nan

    statistic, p_value = mannwhitneyu(pers_data, ctrl_data, alternative='two-sided')
    n1, n2 = len(pers_data), len(ctrl_data)
    r = 2 * statistic / (n1 * n2) - 1
    return p_value, r, statistic

p_values = {}
effect_sizes = {}
u_statistics = {}
for topic in plot_order_types:
    p_val, effect_size, u_stat = perform_mann_whitney_test(delta_total_atomic_claims_df, topic)
    p_values[topic] = p_val
    effect_sizes[topic] = effect_size
    u_statistics[topic] = u_stat

def apply_correction(p_values_dict, correction_method):
    if correction_method is None:
        return p_values_dict

    topics = list(p_values_dict.keys())
    p_vals = np.array([p_values_dict[t] for t in topics])
    n_tests = len(p_vals)

    if correction_method.lower() == 'bonferroni':
        corrected_p = np.minimum(p_vals * n_tests, 1.0)
    elif correction_method.lower() == 'benjamini-hochberg':
        sorted_indices = np.argsort(p_vals)
        sorted_p = p_vals[sorted_indices]
        ranks = np.arange(1, n_tests + 1)
        corrected_p_sorted = np.minimum.accumulate(
            np.minimum(sorted_p * n_tests / ranks, 1.0)[::-1]
        )[::-1]
        corrected_p = np.empty_like(corrected_p_sorted)
        corrected_p[sorted_indices] = corrected_p_sorted
    else:
        raise ValueError(f"Unknown correction method: {correction_method}")

    return {topics[i]: corrected_p[i] for i in range(n_tests)}

corrected_p_values = apply_correction(p_values, USE_CORRECTION)
p_values_to_use = corrected_p_values

# ===========================================
# SPLIT VIOLIN PLOT
# ===========================================
sns.set_style("white")
fig, ax = plt.subplots(figsize=(18, 10))

sns.violinplot(
    data=delta_total_atomic_claims_df,
    x='prompt_label',
    y='delta_total_atomic_claims',
    hue='model_id',
    order=plot_order_labels,
    hue_order=model_order,
    palette=palette,
    split=True,
    inner='quart',       # shows median + IQR lines inside
    linewidth=1.0,
    density_norm='width', # all violins same max width for comparability
    cut=0,               # don't extend density beyond data range
    ax=ax
)

# Make violin edges slightly more visible
for art in ax.collections:
    art.set_edgecolor('white')
    art.set_linewidth(0.8)

# Cosmetics
ax.set_xlabel("")
ax.set_xticklabels(plot_order_labels, fontsize=20)
ax.set_ylabel("Atomic claim difference per response\n(user-preference − counter-preference)",
              fontsize=20, fontweight='bold')
ax.set_title("Net Atomic Claims in Model Responses by Topic",
             fontsize=24, pad=20)

# Y-axis
y_max = 50
step = 10
y_plot_max = y_max + 8
y_plot_min = -y_max - 8
ax.set_ylim(y_plot_min, y_plot_max)

yticks = np.arange(-y_max, y_max + step, step)
ax.set_yticks(yticks)
ax.set_yticklabels([str(int(v)) for v in yticks], fontsize=16)

for y in yticks:
    ax.axhline(y, color="0.85", lw=0.8, ls="--", zorder=0)

# Legend
handles = [
    Patch(facecolor=color_nonpersonal, edgecolor='white', label='Non-personalized Model'),
    Patch(facecolor=color_personalized, edgecolor='white', label='Personalized Model'),
]
ax.legend(handles=handles, title="Model", loc="upper left", bbox_to_anchor=(0.02, 0.21),
          frameon=True, fontsize=14, title_fontsize=14)

# User preferences at top
for i, topic in enumerate(plot_order_types):
    ax.text(i, y_max + 7, user_prefs.get(topic, ""), ha="center", va="top",
            fontsize=16, color="0.2", style='italic')

# Counter-preferences at bottom
for i, topic in enumerate(plot_order_types):
    ax.text(i, -y_max - 1, counter_prefs.get(topic, ""), ha="center", va="top",
            fontsize=16, color="0.2", style='italic')

# Significance stars
for i, topic in enumerate(plot_order_types):
    p_val = p_values_to_use[topic]
    if p_val < 0.001:
        sig_text = "***"
    elif p_val < 0.01:
        sig_text = "**"
    elif p_val < 0.05:
        sig_text = "*"
    else:
        sig_text = ""

    if sig_text:
        ax.text(i, 27, sig_text, ha="center", va="bottom",
                fontsize=22, color="0.2", fontweight='bold')

plt.tight_layout()
fig.savefig(ED_FIG1_OUTPUT, bbox_inches="tight", dpi=300, pad_inches=0.3)
plt.show()

# ===========================================
# PRINT STATISTICS (exact values reported in the legend)
# ===========================================
correction_label = ""
if USE_CORRECTION:
    if USE_CORRECTION.lower() == 'bonferroni':
        correction_label = " (Bonferroni correction)"
    elif USE_CORRECTION.lower() == 'benjamini-hochberg':
        correction_label = " (Benjamini-Hochberg correction)"

ed_fig1_stats = pd.DataFrame([{
    'Topic': type_map.get(t, t).replace('\n', ' '),
    'U': u_statistics[t],
    'raw p': p_values[t],
    'corrected p': p_values_to_use[t],
    'r (rank biserial)': effect_sizes[t],
} for t in plot_order_types])
ed_fig1_stats.to_csv(os.path.join(OUTPUT_DIR, "extended_data_figure_1_stats.csv"), index=False)

print(f"\nStatistical Significance (Mann-Whitney U test{correction_label}):")
print("-" * 110)
print(f"{'Topic':<30} {'U':<10} {'Raw p-value':<15} {'Corrected p':<15} {'Effect Size (r)':<15} {'Sig'}")
print("-" * 110)
for topic in plot_order_types:
    topic_name = type_map.get(topic, topic).replace('\n', ' ')
    raw_p = p_values[topic]
    corr_p = p_values_to_use[topic]
    r = effect_sizes[topic]
    sig = "***" if corr_p < 0.001 else "**" if corr_p < 0.01 else "*" if corr_p < 0.05 else ""
    print(f"{topic_name:<30} {u_statistics[topic]:<10.1f} {raw_p:<15.4g} {corr_p:<15.4g} {r:<15.3f} {sig}")

print("\nEffect Size Interpretation (rank biserial correlation r):")
print("  |r| < 0.10: negligible")
print("  |r| < 0.30: small")
print("  |r| < 0.50: medium")
print("  |r| ≥ 0.50: large")
print("\nNote: Positive r indicates Personalized model has higher delta values (more user-aligned)")
print("      Negative r indicates Non-personalized model has higher delta values")


# =============================================================================
# SOURCE DATA — one workbook per figure (Nature Source Data requirement)
#   Written to <repo>/source_data/
# =============================================================================
SOURCE_DATA_DIR = os.path.join(SCRIPT_DIR, "..", "source_data")
os.makedirs(SOURCE_DATA_DIR, exist_ok=True)

# ---- Figure 1 -----------------------------------------------------------------
fig1a = pd.DataFrame([
    {"Topic": TOPIC_LABELS.get(t, t), "Model condition": m,
     "Fully user-aligned (n)": int(counts[m][i, 0]),
     "Mixed preference (n)": int(counts[m][i, 1]),
     "Fully counter-preference (n)": int(counts[m][i, 2])}
    for i, t in enumerate(topics) for m in (REF, ALT)])
# paired_topic() reads the global SEED; Figure 1 used SEED = 0 (the Figure 2
# section later redefines SEED = 42), so restore it for these values.
_seed_fig2, SEED = SEED, 0
_fig1b_stats = [paired_topic(j) for j in range(len(CATS))]
SEED = _seed_fig2
fig1b = pd.DataFrame([
    {"Response category": CATS[j],
     "Mean paired difference (pp)": _fig1b_stats[j]["diff"],
     "95% CI lower (pp)": _fig1b_stats[j]["lo"],
     "95% CI upper (pp)": _fig1b_stats[j]["hi"]}
    for j in range(len(CATS))])
fig1c = (stats_c['sim_df']
         .rename(columns={'user_preference': 'User-aligned claims',
                          'anti_user_preference': 'Counter-preference claims',
                          'prompt_type': 'Topic', 'model_id': 'Model condition'})
         [['row_id', 'Topic', 'Model condition', 'User-aligned claims', 'Counter-preference claims']])
fig1c['Topic'] = fig1c['Topic'].map(lambda t: TOPIC_LABELS.get(t, t))
with pd.ExcelWriter(os.path.join(SOURCE_DATA_DIR, "Source_Data_Fig1.xlsx")) as xw:
    fig1a.to_excel(xw, sheet_name="Fig1a", index=False)
    fig1b.to_excel(xw, sheet_name="Fig1b", index=False)
    fig1c.to_excel(xw, sheet_name="Fig1c_per_simulation", index=False)
    fig1c_stats.to_excel(xw, sheet_name="Fig1c_tests", index=False)

# ---- Figure 2 -----------------------------------------------------------------
fig2a = pd.DataFrame([{"Model condition": m, "Reasoning-trace category": r,
                       "Text response": o, "n": int(n)}
                      for (m, r, o), n in n_mro.items()])
fig2b = resp[['row_id', 'prompt_type', 'model_id', 'reasoning',
              'user_aligned', 'counter_pref', 'net_diff']].rename(columns={
    'prompt_type': 'Topic', 'model_id': 'Model condition',
    'reasoning': 'Reasoning-trace category', 'user_aligned': 'User-aligned claims',
    'counter_pref': 'Counter-preference claims',
    'net_diff': 'Atomic claim difference score'})
fig2b['Topic'] = fig2b['Topic'].map(lambda t: TOPIC_LABELS.get(t, t))
with pd.ExcelWriter(os.path.join(SOURCE_DATA_DIR, "Source_Data_Fig2.xlsx")) as xw:
    fig2a.to_excel(xw, sheet_name="Fig2a_flows", index=False)
    fig2b.to_excel(xw, sheet_name="Fig2b_per_response", index=False)
    tests.to_excel(xw, sheet_name="Fig2b_tests", index=False)

# ---- Extended Data Figure 1 ----------------------------------------------------
ed1 = delta_total_atomic_claims_df[['row_id', 'prompt_type', 'model_id',
                                    'delta_total_atomic_claims']].rename(columns={
    'prompt_type': 'Topic', 'model_id': 'Model condition',
    'delta_total_atomic_claims': 'Atomic claim difference score'})
ed1['Topic'] = ed1['Topic'].map(lambda t: TOPIC_LABELS.get(t, t))
with pd.ExcelWriter(os.path.join(SOURCE_DATA_DIR, "Source_Data_Extended_Data_Fig1.xlsx")) as xw:
    ed1.to_excel(xw, sheet_name="ED_Fig1_per_response", index=False)
    ed_fig1_stats.to_excel(xw, sheet_name="ED_Fig1_tests", index=False)

print(f"\nSource data written to {os.path.abspath(SOURCE_DATA_DIR)}")
