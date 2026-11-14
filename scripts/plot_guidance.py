"""Miss-distance distribution figures for P8.3/P8.7."""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

ROOT = Path(__file__).resolve().parents[1]
TABLES = ROOT / "results" / "tables"
FIGS = ROOT / "results" / "figures"


def main() -> int:
    import pandas as pd
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    path = TABLES / "guidance_distributions.csv"
    if not path.exists():
        print("no raw distribution table; run collect_guidance_distributions.py")
        return 1
    df = pd.read_csv(path)
    FIGS.mkdir(parents=True, exist_ok=True)
    for platform, group in df.groupby("platform"):
        behaviors = sorted(group["behavior"].unique())
        fig, axes = plt.subplots(4, 4, figsize=(15, 11), sharey=True)
        for ax, behavior in zip(axes.ravel(), behaviors):
            data, labels = [], []
            for arm in ("apn_raw", "apn_filtered"):
                vals = group[(group.behavior == behavior) &
                             (group.arm == arm)]["miss"].to_numpy()
                data.append(vals)
                labels.append("A1" if arm == "apn_raw" else "A2")
            ax.boxplot(data, labels=labels, showfliers=False)
            ax.set_title(behavior, fontsize=8)
            ax.tick_params(labelsize=7)
            ax.grid(alpha=.25)
        fig.suptitle(f"{platform}: miss-distance distributions (100 seeds per cell)")
        fig.tight_layout()
        out = FIGS / f"guidance_{platform}_miss_distribution.png"
        fig.savefig(out, dpi=150)
        plt.close(fig)
        print("wrote", out)
    return 0


if __name__ == "__main__":
    sys.exit(main())
