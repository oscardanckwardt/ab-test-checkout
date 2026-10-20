"""
A/B test on checkout conversion – a small, reproducible Python experiment.

Scenario: an e-commerce merchant tests a new version of the payment step (variant B)
against today's version (variant A). The data is *simulated* (no real customer data),
but the analysis is the one used in practice:

  1. Conversion rate per variant with 95 % confidence intervals (Wilson)
  2. Two-sided z-test for two proportions (p-value)
  3. Confidence interval for the difference – absolute and relative uplift
  4. Segmented analysis (mobile / desktop) – to catch effects that differ by segment
  5. Power / MDE: how small an effect can the test detect with the traffic we have?
  6. Chart: daily and cumulative conversion per variant

Run:  python ab_test_checkout.py
Deps: pandas, numpy, scipy, matplotlib  (pip install -r requirements.txt)
"""
from __future__ import annotations

from dataclasses import dataclass

import matplotlib
matplotlib.use("Agg")  # headless rendering; remove this line if you want an interactive window
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy import stats

# ---------------------------------------------------------------------------
# 1. Simulate checkout sessions
# ---------------------------------------------------------------------------

def simulate_sessions(
    n_days: int = 14,
    sessions_per_day: int = 3000,
    mobile_share: float = 0.65,
    base_rate: dict[str, float] | None = None,
    uplift_b: dict[str, float] | None = None,
    seed: int = 2026,
) -> pd.DataFrame:
    """Return one row per checkout session: date, variant, device, converted (0/1).

    Conversion depends on device (mobile converts worse than desktop) and on the
    variant: B gets an additive uplift in percentage points per device.
    """
    base_rate = base_rate or {"mobile": 0.77, "desktop": 0.86}
    uplift_b = uplift_b or {"mobile": 0.020, "desktop": 0.005}
    rng = np.random.default_rng(seed)

    n = n_days * sessions_per_day
    dates = np.repeat(pd.date_range("2026-09-01", periods=n_days, freq="D"), sessions_per_day)
    variant = rng.choice(["A", "B"], size=n)                      # 50/50 randomisation
    device = np.where(rng.random(n) < mobile_share, "mobile", "desktop")

    p = np.array([base_rate[d] for d in device])
    p = p + np.where(variant == "B", [uplift_b[d] for d in device], 0.0)
    converted = (rng.random(n) < p).astype(int)

    return pd.DataFrame({"date": dates, "variant": variant, "device": device, "converted": converted})


# ---------------------------------------------------------------------------
# 2–3. Statistics: Wilson CI, two-proportion z-test, CI for the difference
# ---------------------------------------------------------------------------

def wilson_ci(successes: int, n: int, alpha: float = 0.05) -> tuple[float, float]:
    """Wilson score interval for a proportion (better than the plain Wald interval)."""
    if n == 0:
        return (np.nan, np.nan)
    z = stats.norm.ppf(1 - alpha / 2)
    p = successes / n
    denom = 1 + z**2 / n
    centre = (p + z**2 / (2 * n)) / denom
    half = z * np.sqrt(p * (1 - p) / n + z**2 / (4 * n**2)) / denom
    return (centre - half, centre + half)


@dataclass
class ZTestResult:
    rate_a: float
    rate_b: float
    diff: float            # absolute difference B − A (percentage points as a fraction)
    rel_uplift: float      # relative uplift B vs A
    z: float
    p_value: float
    ci_diff: tuple[float, float]

    def __str__(self) -> str:
        lo, hi = self.ci_diff
        return (
            f"A: {self.rate_a:.2%}   B: {self.rate_b:.2%}   diff: {self.diff*100:+.2f} pp "
            f"(95 % CI {lo*100:+.2f} to {hi*100:+.2f} pp)   relative uplift: {self.rel_uplift:+.2%}\n"
            f"z = {self.z:.2f}, p = {self.p_value:.4f} → "
            + ("statistically significant at the 5 % level" if self.p_value < 0.05 else "NOT significant at the 5 % level")
        )


def ztest_two_proportions(conv_a: int, n_a: int, conv_b: int, n_b: int, alpha: float = 0.05) -> ZTestResult:
    """Two-sided z-test for H0: p_A == p_B, plus an (unpooled) CI for the difference."""
    p_a, p_b = conv_a / n_a, conv_b / n_b
    p_pool = (conv_a + conv_b) / (n_a + n_b)
    se_pool = np.sqrt(p_pool * (1 - p_pool) * (1 / n_a + 1 / n_b))
    z = (p_b - p_a) / se_pool
    p_value = 2 * (1 - stats.norm.cdf(abs(z)))

    se_diff = np.sqrt(p_a * (1 - p_a) / n_a + p_b * (1 - p_b) / n_b)
    zc = stats.norm.ppf(1 - alpha / 2)
    diff = p_b - p_a
    return ZTestResult(p_a, p_b, diff, diff / p_a, z, p_value, (diff - zc * se_diff, diff + zc * se_diff))


# ---------------------------------------------------------------------------
# 5. Power analysis: minimum detectable effect (MDE) and required sample size
# ---------------------------------------------------------------------------

def mde(n_per_variant: int, p_base: float, alpha: float = 0.05, power: float = 0.80) -> float:
    """Smallest absolute difference (fraction) detectable with the given sample size."""
    z_alpha = stats.norm.ppf(1 - alpha / 2)
    z_beta = stats.norm.ppf(power)
    return (z_alpha + z_beta) * np.sqrt(2 * p_base * (1 - p_base) / n_per_variant)


def sample_size(p_base: float, mde_abs: float, alpha: float = 0.05, power: float = 0.80) -> int:
    """Sessions needed *per variant* to detect an absolute difference of `mde_abs`."""
    z_alpha = stats.norm.ppf(1 - alpha / 2)
    z_beta = stats.norm.ppf(power)
    return int(np.ceil(2 * p_base * (1 - p_base) * ((z_alpha + z_beta) / mde_abs) ** 2))


def mde_table(p_base: float, sizes=(5_000, 10_000, 20_000, 50_000, 100_000)) -> pd.DataFrame:
    rows = [{"sessions per variant": n, "MDE (pp)": mde(n, p_base) * 100, "MDE (relative)": mde(n, p_base) / p_base} for n in sizes]
    return pd.DataFrame(rows)


# ---------------------------------------------------------------------------
# 4. Summaries
# ---------------------------------------------------------------------------

def summarise(df: pd.DataFrame, by: list[str]) -> pd.DataFrame:
    """Sessions, conversions, conversion rate and Wilson CI per group."""
    g = df.groupby(by, observed=True)["converted"].agg(sessions="size", conversions="sum").reset_index()
    g["rate"] = g["conversions"] / g["sessions"]
    ci = g.apply(lambda r: wilson_ci(int(r["conversions"]), int(r["sessions"])), axis=1)
    g["ci_low"] = [c[0] for c in ci]
    g["ci_high"] = [c[1] for c in ci]
    return g


# ---------------------------------------------------------------------------
# 6. Chart
# ---------------------------------------------------------------------------

def plot(df: pd.DataFrame, path: str = "ab_test_checkout.png") -> str:
    daily = df.groupby(["date", "variant"])["converted"].agg(["size", "sum"]).reset_index()
    daily["rate"] = daily["sum"] / daily["size"]
    daily = daily.sort_values("date")
    daily["cum_n"] = daily.groupby("variant")["size"].cumsum()
    daily["cum_conv"] = daily.groupby("variant")["sum"].cumsum()
    daily["cum_rate"] = daily["cum_conv"] / daily["cum_n"]
    ci = daily.apply(lambda r: wilson_ci(int(r["cum_conv"]), int(r["cum_n"])), axis=1)
    daily["cum_low"] = [c[0] for c in ci]
    daily["cum_high"] = [c[1] for c in ci]

    colours = {"A": "#8a96a6", "B": "#1d4ed8"}
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.2))
    for v, d in daily.groupby("variant"):
        axes[0].plot(d["date"], d["rate"] * 100, marker="o", ms=3, color=colours[v], label=f"Variant {v}")
        axes[1].plot(d["date"], d["cum_rate"] * 100, color=colours[v], label=f"Variant {v}")
        axes[1].fill_between(d["date"], d["cum_low"] * 100, d["cum_high"] * 100, color=colours[v], alpha=0.15)
    axes[0].set_title("Daily conversion rate")
    axes[1].set_title("Cumulative conversion rate with 95 % CI")
    for ax in axes:
        ax.set_ylabel("Conversion (%)")
        ax.grid(alpha=0.3)
        ax.legend(frameon=False)
        ax.tick_params(axis="x", labelrotation=30)
    fig.suptitle("A/B test – checkout payment step (simulated data)", fontweight="bold")
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)
    return path


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    pd.set_option("display.width", 120)
    df = simulate_sessions()
    print(f"Simulated {len(df):,} checkout sessions over {df['date'].nunique()} days\n")

    overall = summarise(df, ["variant"])
    print("Conversion per variant (Wilson 95 % CI):")
    print(overall.to_string(index=False, formatters={"rate": "{:.2%}".format, "ci_low": "{:.2%}".format, "ci_high": "{:.2%}".format}), "\n")

    a = overall.set_index("variant").loc["A"]
    b = overall.set_index("variant").loc["B"]
    result = ztest_two_proportions(int(a["conversions"]), int(a["sessions"]), int(b["conversions"]), int(b["sessions"]))
    print("Two-proportion z-test:")
    print(result, "\n")

    seg = summarise(df, ["device", "variant"])
    print("Segmented by device:")
    print(seg.to_string(index=False, formatters={"rate": "{:.2%}".format, "ci_low": "{:.2%}".format, "ci_high": "{:.2%}".format}), "\n")
    for device, d in seg.groupby("device"):
        d = d.set_index("variant")
        r = ztest_two_proportions(int(d.loc["A", "conversions"]), int(d.loc["A", "sessions"]), int(d.loc["B", "conversions"]), int(d.loc["B", "sessions"]))
        print(f"  {device:8s} diff {r.diff*100:+.2f} pp, p = {r.p_value:.4f}")
    print()

    p_base = float(a["rate"])
    print(f"Power analysis (alpha = 5 %, power = 80 %, baseline {p_base:.1%}):")
    print(mde_table(p_base).to_string(index=False, formatters={"MDE (pp)": "{:.2f}".format, "MDE (relative)": "{:.2%}".format}))
    print(f"\nTo detect +1.0 pp you need {sample_size(p_base, 0.01):,} sessions per variant; "
          f"this test had {int(a['sessions']):,} per variant → MDE {mde(int(a['sessions']), p_base)*100:.2f} pp.\n")

    path = plot(df)
    print(f"Chart saved to {path}")


if __name__ == "__main__":
    main()
