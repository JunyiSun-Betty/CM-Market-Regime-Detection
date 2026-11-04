"""Charts for the report (PNG files)."""
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker

INK, MUTED, GRID, SURFACE = "#0b0b0b", "#52514e", "#e6e5e0", "#fcfcfb"
BLUE, ORANGE, AQUA, YELLOW = "#2a78d6", "#eb6834", "#1baf7a", "#eda100"


def _style(ax):
    ax.set_facecolor(SURFACE)
    ax.grid(axis="y", color=GRID, linewidth=0.8)
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
    for s in ("left", "bottom"):
        ax.spines[s].set_color(GRID)
    ax.tick_params(colors=MUTED, labelsize=9)


def plot_regimes(close, p_high, regime, path, title, truth=None):
    idx = p_high.index
    high = (regime.reindex(idx) == "high_vol").to_numpy()
    fig, (a1, a2) = plt.subplots(2, 1, figsize=(11, 6.2), sharex=True,
                                 gridspec_kw={"height_ratios": [2, 1]})
    fig.patch.set_facecolor(SURFACE)
    for ax in (a1, a2):
        _style(ax)
    a1.fill_between(idx, 0, 1, where=high, transform=a1.get_xaxis_transform(),
                    color=ORANGE, alpha=0.22, linewidth=0, label="High-volatility regime (model)")
    a1.plot(idx, close.reindex(idx), color=BLUE, linewidth=1.5, label=f"{close.name} price")
    a1.set_yscale("log")
    a1.yaxis.set_major_locator(mticker.LogLocator(subs=(1.0, 2.0, 5.0)))
    a1.yaxis.set_major_formatter(mticker.FuncFormatter(lambda v, _: f"{v:,.0f}"))
    a1.yaxis.set_minor_formatter(mticker.NullFormatter())
    a1.set_ylabel("Price (log scale)", color=MUTED, fontsize=9)
    a1.legend(loc="upper left", frameon=False, fontsize=9, labelcolor=INK)
    a1.set_title(title, loc="left", color=INK, fontsize=12, fontweight="bold")
    if truth is not None:
        tr = truth.reindex(idx).fillna(0).to_numpy().astype(bool)
        a2.fill_between(idx, 0, 1, where=tr, color=GRID, linewidth=0, label="True stress (simulation)")
    a2.plot(idx, p_high, color=ORANGE, linewidth=1.3, label="P(high volatility)")
    a2.axhline(0.5, color=MUTED, linewidth=0.8, linestyle="--")
    a2.set_ylim(0, 1)
    a2.set_ylabel("P(high vol)", color=MUTED, fontsize=9)
    a2.legend(loc="lower left", bbox_to_anchor=(0, 1.0), frameon=False, fontsize=9,
              labelcolor=INK, ncol=2)
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


def plot_clusters(model, path):
    """The last fit: each dot is one past day, placed by its first two
    principal components and coloured by the regime the clustering gave it."""
    Z, lab = model.Z_train, model.labels_train
    fig, ax = plt.subplots(figsize=(6.4, 5))
    fig.patch.set_facecolor(SURFACE)
    _style(ax)
    ax.grid(axis="x", color=GRID, linewidth=0.8)
    for k, colour, name in ((1 - model.high, BLUE, "Calm"), (model.high, ORANGE, "Highly volatile")):
        m = lab == k
        ax.scatter(Z[m, 0], Z[m, 1], s=9, color=colour, alpha=0.55, linewidths=0,
                   label=f"{name} ({m.mean():.0%} of days)")
    ax.scatter(model.centres[:, 0], model.centres[:, 1], s=90, color=INK, marker="x",
               linewidths=2, label="Cluster centres")
    ax.set_xlabel("Principal component 1", color=MUTED, fontsize=9)
    ax.set_ylabel("Principal component 2", color=MUTED, fontsize=9)
    ax.set_title("How the clustering splits past days (last fit)", loc="left",
                 color=INK, fontsize=11, fontweight="bold")
    ax.legend(frameon=False, fontsize=9, labelcolor=INK, loc="best")
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


def plot_strategies(curves, path, title):
    fig, ax = plt.subplots(figsize=(11, 4.6))
    fig.patch.set_facecolor(SURFACE)
    _style(ax)
    for (name, s), colour in zip(curves.items(), (MUTED, BLUE, ORANGE, AQUA)):
        ax.plot(s.index, s, color=colour, linewidth=1.5, label=name)
    ax.set_yscale("log")
    ax.yaxis.set_major_locator(mticker.LogLocator(subs=(1.0, 2.0, 5.0)))
    ax.yaxis.set_major_formatter(mticker.FuncFormatter(lambda v, _: f"{v:g}x"))
    ax.yaxis.set_minor_formatter(mticker.NullFormatter())
    ax.set_ylabel("Growth of 1 (log scale)", color=MUTED, fontsize=9)
    ax.set_title(title, loc="left", color=INK, fontsize=12, fontweight="bold")
    ax.legend(loc="upper left", frameon=False, fontsize=9, labelcolor=INK, ncol=2)
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)
