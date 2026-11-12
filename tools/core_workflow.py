"""Generate the q-CDMA-CV-QKD core workflow diagram."""

from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch  # noqa: E402


ROOT = Path(__file__).resolve().parents[1]
OUTPUT = ROOT / "results" / "figures" / "core_workflow.png"


def box(ax, x: float, y: float, width: float, height: float, text: str,
        color: str, fontsize: float = 9) -> None:
    patch = FancyBboxPatch(
        (x, y), width, height,
        boxstyle="round,pad=0.02,rounding_size=0.025",
        facecolor=color, edgecolor="#263238", linewidth=1.2,
    )
    ax.add_patch(patch)
    ax.text(x + width / 2, y + height / 2, text, ha="center", va="center",
            fontsize=fontsize, color="#17202a", linespacing=1.25)


def arrow(ax, start: tuple[float, float], end: tuple[float, float],
          color: str = "#455a64", connectionstyle: str = "arc3") -> None:
    ax.add_patch(FancyArrowPatch(
        start, end, arrowstyle="-|>", mutation_scale=13, linewidth=1.4,
        color=color, connectionstyle=connectionstyle,
    ))


def main() -> Path:
    fig, ax = plt.subplots(figsize=(15, 8), dpi=160)
    ax.set_xlim(0, 15)
    ax.set_ylim(0, 8)
    ax.axis("off")
    fig.patch.set_facecolor("white")

    ax.text(7.5, 7.65, "q-CDMA-CV-QKD core workflow",
            ha="center", va="center", fontsize=18, fontweight="bold",
            color="#102a43")
    ax.text(7.5, 7.25,
            "Bootstrap once, then repeat authenticated coded CV-QKD rounds",
            ha="center", va="center", fontsize=10, color="#52606d")

    # Bootstrap path.
    box(ax, 0.35, 5.85, 2.0, 0.75, "Pre-shared root\nK0 / root ring",
        "#e3f2fd")
    box(ax, 3.0, 5.85, 2.15, 0.75, "Bootstrap split\nK0 -> a1, c1",
        "#e8eaf6")
    box(ax, 5.85, 5.85, 2.35, 0.75, "Initial codebook\nPRF(c1, frame, user)",
        "#e0f2f1")
    arrow(ax, (2.35, 6.225), (3.0, 6.225))
    arrow(ax, (5.15, 6.225), (5.85, 6.225))
    ax.text(1.35, 6.8, "one-time setup", ha="center", fontsize=9,
            color="#607d8b", style="italic")

    # Main round.
    box(ax, 0.35, 3.35, 2.15, 1.0, "1. Authenticate\ncode update + sync\nwith a_n",
        "#fff3e0")
    box(ax, 3.05, 3.35, 2.15, 1.0, "2. Generate / refresh\ncodebook locally\nwith PRF(c_n)",
        "#e0f2f1")
    box(ax, 5.75, 3.35, 2.15, 1.0, "3. Coded quantum\ntransmission\n+ multi-user CDMA",
        "#e8f5e9")
    box(ax, 8.45, 3.35, 2.15, 1.0, "4. Parameter\nestimation\n(PE disclosure)",
        "#f3e5f5")
    box(ax, 11.15, 3.35, 2.15, 1.0, "5. Reconcile +\nprivacy amplify\n-> K_n",
        "#ede7f6")

    for x in (2.5, 5.2, 7.9, 10.6):
        arrow(ax, (x, 3.85), (x + 0.55, 3.85))

    # Checks and failure handling.
    box(ax, 8.45, 1.35, 2.15, 0.85, "Checks pass?\nauth / PE / budget",
        "#fff8e1")
    arrow(ax, (9.525, 3.35), (9.525, 2.2))
    ax.text(9.78, 2.68, "validate", fontsize=8, color="#607d8b")
    box(ax, 11.15, 1.35, 2.15, 0.85, "Abort safely\n(no seed release)",
        "#ffebee")
    arrow(ax, (10.6, 1.775), (11.15, 1.775), color="#c62828")
    ax.text(10.85, 1.98, "no", fontsize=8, color="#c62828")

    # Successful KDF split and loop.
    box(ax, 5.75, 1.35, 2.15, 0.85, "KDF domain split\nK_n -> a_(n+1),\nc_(n+1), u_n",
        "#e8eaf6", fontsize=8.5)
    arrow(ax, (8.45, 1.775), (7.9, 1.775), color="#2e7d32")
    ax.text(8.06, 1.98, "yes", fontsize=8, color="#2e7d32")
    arrow(ax, (6.825, 2.2), (6.825, 3.35), color="#2e7d32")
    ax.text(7.0, 2.72, "next round", fontsize=8, color="#2e7d32")

    # Feedback arrows into the next round's inputs.
    arrow(ax, (6.825, 2.2), (1.43, 2.2), color="#2e7d32",
          connectionstyle="angle3,angleA=90,angleB=0")
    arrow(ax, (1.43, 2.2), (1.43, 3.35), color="#2e7d32")
    ax.text(3.9, 2.38, "a_(n+1) authenticates the next update; "
            "c_(n+1) drives the next PRF codebook",
            ha="center", va="center", fontsize=8.5, color="#2e7d32")

    # Legend / model boundary.
    ax.text(0.35, 0.55,
            "Model boundary: K_n is produced by the finite-size CV-QKD rate; "
            "the loop does not improve the underlying quantum security bound.",
            fontsize=9, color="#455a64")
    ax.text(0.35, 0.2,
            "Default code model: computational PRF security with full refresh (f = 1).",
            fontsize=9, color="#455a64")

    OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(OUTPUT, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    print(f"figure -> {OUTPUT}")
    return OUTPUT


if __name__ == "__main__":
    main()
