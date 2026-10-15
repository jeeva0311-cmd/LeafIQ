"""
LeafIQ - Phase 4: Benchmarking
================================
Compares all trained models using existing evaluation_results.csv and
training CSVs. Never retrains — loads only what is already saved.

Outputs (outputs/plots/):
  benchmark_comparison.png      — 4-panel bar chart (acc, F1, prec, rec)
  benchmark_params_vs_acc.png   — scatter: param count vs accuracy
  benchmark_inference.png       — horizontal bar: inference time
  benchmark_training_overlay.png — all training curves on one axes

Outputs (outputs/):
  benchmark_summary.csv         — full comparison table (all metrics)
  benchmark_summary.txt         — human-readable table

Usage:
    python src/benchmark.py
"""

import os, sys, csv, logging
os.environ["TF_ENABLE_ONEDNN_OPTS"] = "0"
os.environ["TF_CPP_MIN_LOG_LEVEL"]  = "3"

import numpy as np
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import seaborn as sns

PROJECT_ROOT = Path(__file__).resolve().parent.parent
OUTPUTS      = PROJECT_ROOT / "outputs"
PLOTS        = OUTPUTS / "plots"
LOG_DIR      = OUTPUTS / "logs"

PLOTS.mkdir(parents=True, exist_ok=True)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-8s  %(message)s",
    handlers=[
        logging.FileHandler(LOG_DIR / "benchmark.log", encoding="utf-8"),
        logging.StreamHandler(sys.stdout),
    ],
)
log = logging.getLogger(__name__)

CLASS_NAMES = ["Tomato_Healthy", "Tomato_Early_Blight",
               "Tomato_Late_Blight", "Tomato_Leaf_Mold"]

DISPLAY = {
    "custom_cnn":     "Custom CNN",
    "mobilenetv2":    "MobileNetV2-A",
    "mobilenetv2_ft": "MobileNetV2-FT",
}

COLORS = {
    "custom_cnn":     "#2ecc71",
    "mobilenetv2":    "#3498db",
    "mobilenetv2_ft": "#e74c3c",
}

TRAIN_CSV = {
    "custom_cnn":     LOG_DIR / "train_custom_cnn.csv",
    "mobilenetv2":    LOG_DIR / "train_mobilenetv2.csv",
    "mobilenetv2_ft": LOG_DIR / "train_mobilenetv2_ft.csv",
}


# ─────────────────────────────────────────────────────────────────────────────
# Data loading
# ─────────────────────────────────────────────────────────────────────────────

def load_eval_results() -> list[dict]:
    path = OUTPUTS / "evaluation_results.csv"
    if not path.exists():
        raise FileNotFoundError(f"evaluation_results.csv not found: {path}")
    rows = []
    with open(path, newline="", encoding="utf-8") as f:
        for r in csv.DictReader(f):
            rows.append(r)
    log.info("Loaded %d model entries from evaluation_results.csv", len(rows))
    return rows


def load_train_history(model_key: str) -> dict[str, list]:
    path = TRAIN_CSV.get(model_key)
    if not path or not path.exists():
        return {}
    epochs, acc, val_acc, loss, val_loss = [], [], [], [], []
    with open(path, newline="", encoding="utf-8") as f:
        for row in csv.DictReader(f):
            if not row.get("epoch", "").strip().lstrip("-").isdigit():
                continue
            epochs.append(int(row["epoch"]) + 1)
            acc.append(float(row["accuracy"]))
            val_acc.append(float(row["val_accuracy"]))
            loss.append(float(row["loss"]))
            val_loss.append(float(row["val_loss"]))
    return {"epochs": epochs, "acc": acc, "val_acc": val_acc,
            "loss": loss, "val_loss": val_loss}


# ─────────────────────────────────────────────────────────────────────────────
# Charts
# ─────────────────────────────────────────────────────────────────────────────

def _pct(val_str: str) -> float | None:
    try:
        return float(val_str) * 100 if val_str else None
    except (ValueError, TypeError):
        return None


def plot_comparison_bars(rows: list[dict]) -> Path:
    """4-panel grouped bar: accuracy / F1 / precision / recall."""
    valid   = [r for r in rows if r["status"] == "OK"]
    failed  = [r for r in rows if r["status"] != "OK"]
    names   = [DISPLAY.get(r["model_key"], r["model_key"]) for r in valid]
    cols    = [COLORS.get(r["model_key"], "#95a5a6") for r in valid]

    metrics = [
        ("accuracy",         "Test Accuracy (%)"),
        ("f1_macro",         "Macro F1 (%)"),
        ("precision_macro",  "Macro Precision (%)"),
        ("recall_macro",     "Macro Recall (%)"),
    ]

    fig, axes = plt.subplots(2, 2, figsize=(13, 9))
    axes = axes.flatten()
    fig.suptitle("LeafIQ — Model Benchmark Comparison\n(test set, n=821)",
                 fontsize=13, fontweight="bold", y=1.01)

    for ax, (key, label) in zip(axes, metrics):
        vals = [_pct(r.get(key, "")) for r in valid]
        bars = ax.bar(names, vals, color=cols, alpha=0.88, edgecolor="white",
                      linewidth=0.8, width=0.5)
        for bar, v in zip(bars, vals):
            if v is not None:
                ax.text(bar.get_x() + bar.get_width() / 2,
                        bar.get_height() + 0.4,
                        f"{v:.2f}%", ha="center", va="bottom",
                        fontsize=9, fontweight="bold")
        ax.set_title(label, fontsize=11)
        ax.set_ylim(0, 110)
        ax.set_ylabel("Score (%)")
        ax.grid(axis="y", alpha=0.3, linestyle="--")
        ax.tick_params(axis="x", labelsize=9)
        sns.despine(ax=ax)

    # Annotate failed models
    if failed:
        note = "\n".join(
            f"{DISPLAY.get(r['model_key'], r['model_key'])}: {r['status']}"
            for r in failed
        )
        for ax in axes:
            ax.text(0.99, 0.03, f"Not evaluated:\n{note}",
                    transform=ax.transAxes, ha="right", va="bottom",
                    fontsize=7, color="#c0392b",
                    bbox=dict(boxstyle="round,pad=0.3", fc="white",
                              ec="#c0392b", alpha=0.8))

    plt.tight_layout()
    p = PLOTS / "benchmark_comparison.png"
    fig.savefig(p, dpi=150, bbox_inches="tight")
    plt.close(fig)
    log.info("Saved: %s", p)
    return p


def plot_params_vs_accuracy(rows: list[dict]) -> Path:
    """Scatter: param count (log-scale x) vs test accuracy (y)."""
    fig, ax = plt.subplots(figsize=(8, 5))
    for r in rows:
        name = DISPLAY.get(r["model_key"], r["model_key"])
        col  = COLORS.get(r["model_key"], "#95a5a6")
        if r["status"] == "OK" and r.get("n_params") and r.get("accuracy"):
            params = int(r["n_params"])
            acc    = float(r["accuracy"]) * 100
            ax.scatter(params, acc, s=180, color=col, zorder=5,
                       edgecolors="white", linewidths=1.5)
            ax.annotate(f" {name}\n {acc:.2f}%",
                        (params, acc), fontsize=8,
                        va="center", xytext=(8, 0),
                        textcoords="offset points")
        else:
            if r.get("n_params"):
                params = int(r["n_params"])
                ax.scatter(params, 0, s=180, color=col, zorder=5,
                           marker="x", linewidths=2)
                ax.annotate(f" {name}\n (failed)",
                            (params, 0), fontsize=8, color="#c0392b",
                            va="bottom", xytext=(8, 4),
                            textcoords="offset points")

    ax.set_xscale("log")
    ax.set_xlabel("Parameter Count (log scale)", fontsize=11)
    ax.set_ylabel("Test Accuracy (%)", fontsize=11)
    ax.set_title("Model Efficiency: Parameters vs Accuracy", fontsize=12)
    ax.set_ylim(-5, 105)
    ax.grid(alpha=0.3, linestyle="--")
    ax.axhline(y=98.17, color="#2ecc71", linestyle=":", alpha=0.5,
               label="CNN baseline 98.17%")
    ax.legend(fontsize=8)
    sns.despine(ax=ax)
    plt.tight_layout()
    p = PLOTS / "benchmark_params_vs_acc.png"
    fig.savefig(p, dpi=150, bbox_inches="tight")
    plt.close(fig)
    log.info("Saved: %s", p)
    return p


def plot_inference_time(rows: list[dict]) -> Path:
    """Horizontal bar: inference time per image (ms)."""
    valid = [r for r in rows if r["status"] == "OK" and r.get("inference_ms")]
    names = [DISPLAY.get(r["model_key"], r["model_key"]) for r in valid]
    times = [float(r["inference_ms"]) for r in valid]
    cols  = [COLORS.get(r["model_key"], "#95a5a6") for r in valid]

    fig, ax = plt.subplots(figsize=(8, max(3, len(valid) * 1.4)))
    bars = ax.barh(names, times, color=cols, alpha=0.88,
                   edgecolor="white", linewidth=0.8, height=0.5)
    for bar, t in zip(bars, times):
        ax.text(bar.get_width() + 2, bar.get_y() + bar.get_height() / 2,
                f"{t:.1f} ms", va="center", fontsize=10, fontweight="bold")
    ax.set_xlabel("Inference Time per Image (ms, CPU)", fontsize=11)
    ax.set_title("Inference Speed Comparison\n(CPU, single image, TF 2.18)", fontsize=12)
    ax.set_xlim(0, max(times) * 1.25)
    ax.grid(axis="x", alpha=0.3, linestyle="--")
    sns.despine(ax=ax)
    plt.tight_layout()
    p = PLOTS / "benchmark_inference.png"
    fig.savefig(p, dpi=150, bbox_inches="tight")
    plt.close(fig)
    log.info("Saved: %s", p)
    return p


def plot_training_overlay(rows: list[dict]) -> Path:
    """All val_accuracy curves overlaid on one chart for direct comparison."""
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(13, 5))
    fig.suptitle("Training History — All Models", fontsize=13,
                 fontweight="bold")

    any_plotted = False
    for r in rows:
        key  = r["model_key"]
        hist = load_train_history(key)
        if not hist:
            continue
        name = DISPLAY.get(key, key)
        col  = COLORS.get(key, "#95a5a6")
        lbl  = name if r["status"] == "OK" else f"{name} (failed)"
        ls   = "-" if r["status"] == "OK" else "--"

        ax1.plot(hist["epochs"], [v * 100 for v in hist["val_acc"]],
                 color=col, linestyle=ls, linewidth=2, marker="o",
                 markersize=3, label=lbl, alpha=0.9)
        ax1.plot(hist["epochs"], [v * 100 for v in hist["acc"]],
                 color=col, linestyle=":", linewidth=1, alpha=0.5)

        ax2.plot(hist["epochs"], hist["val_loss"],
                 color=col, linestyle=ls, linewidth=2, marker="o",
                 markersize=3, label=lbl, alpha=0.9)
        ax2.plot(hist["epochs"], hist["loss"],
                 color=col, linestyle=":", linewidth=1, alpha=0.5)
        any_plotted = True

    for ax, ylabel, title in [
        (ax1, "Accuracy (%)", "Validation Accuracy (solid) / Train Accuracy (dotted)"),
        (ax2, "Loss",         "Validation Loss (solid) / Train Loss (dotted)"),
    ]:
        ax.set_xlabel("Epoch"); ax.set_ylabel(ylabel)
        ax.set_title(title, fontsize=10)
        ax.legend(fontsize=8)
        ax.grid(alpha=0.3, linestyle="--")
        sns.despine(ax=ax)

    if not any_plotted:
        log.warning("No training history found to plot.")

    plt.tight_layout()
    p = PLOTS / "benchmark_training_overlay.png"
    fig.savefig(p, dpi=150, bbox_inches="tight")
    plt.close(fig)
    log.info("Saved: %s", p)
    return p


# ─────────────────────────────────────────────────────────────────────────────
# Summary table
# ─────────────────────────────────────────────────────────────────────────────

def save_benchmark_csv(rows: list[dict]) -> Path:
    fields = [
        "model_key", "display_name", "status",
        "accuracy_pct", "f1_macro_pct", "precision_macro_pct",
        "recall_macro_pct", "inference_ms", "n_params",
        "params_per_accuracy", "best_val_acc_training_pct", "note",
    ]
    out_rows = []
    for r in rows:
        acc  = _pct(r.get("accuracy", ""))
        f1   = _pct(r.get("f1_macro", ""))
        prec = _pct(r.get("precision_macro", ""))
        rec  = _pct(r.get("recall_macro", ""))
        params = int(r["n_params"]) if r.get("n_params") else None
        bva  = _pct(r.get("best_val_acc_training", ""))
        eff  = round(params / acc, 0) if (params and acc) else None
        out_rows.append({
            "model_key":                  r["model_key"],
            "display_name":               DISPLAY.get(r["model_key"], r["model_key"]),
            "status":                     r["status"],
            "accuracy_pct":               round(acc, 4)  if acc  is not None else "N/A",
            "f1_macro_pct":               round(f1, 4)   if f1   is not None else "N/A",
            "precision_macro_pct":        round(prec, 4) if prec is not None else "N/A",
            "recall_macro_pct":           round(rec, 4)  if rec  is not None else "N/A",
            "inference_ms":               r.get("inference_ms", "N/A"),
            "n_params":                   params if params else "N/A",
            "params_per_accuracy":        eff if eff else "N/A",
            "best_val_acc_training_pct":  round(bva, 4) if bva is not None else "N/A",
            "note":                       r.get("note", ""),
        })
    path = OUTPUTS / "benchmark_summary.csv"
    with open(path, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader(); w.writerows(out_rows)
    log.info("Saved: %s", path)
    return path


def save_benchmark_txt(rows: list[dict]) -> Path:
    lines = [
        "LeafIQ - Phase 4 Benchmark Summary",
        "=" * 72,
        f"{'Model':<26} {'Status':<18} {'Acc%':>7} {'F1%':>7} "
        f"{'Prec%':>7} {'Rec%':>7} {'ms/img':>7} {'Params':>9}",
        "-" * 72,
    ]
    for r in rows:
        acc  = f"{_pct(r.get('accuracy','')):.2f}" if _pct(r.get('accuracy','')) else "N/A"
        f1   = f"{_pct(r.get('f1_macro','')):.2f}" if _pct(r.get('f1_macro','')) else "N/A"
        prec = f"{_pct(r.get('precision_macro','')):.2f}" if _pct(r.get('precision_macro','')) else "N/A"
        rec  = f"{_pct(r.get('recall_macro','')):.2f}" if _pct(r.get('recall_macro','')) else "N/A"
        ms   = r.get('inference_ms', 'N/A')
        par  = f"{int(r['n_params']):,}" if r.get('n_params') else "N/A"
        name = DISPLAY.get(r["model_key"], r["model_key"])
        lines.append(
            f"{name:<26} {r['status']:<18} {acc:>7} {f1:>7} "
            f"{prec:>7} {rec:>7} {ms:>7} {par:>9}"
        )
    lines += ["=" * 72,
              "",
              "Notes:",
              "  - All metrics on held-out test set (821 images, never seen during training)",
              "  - Inference time measured on CPU (single-image batch, TensorFlow 2.18)",
              "  - N/A = model did not converge; evaluation skipped to avoid fabricated results",
              "  - Custom CNN trained from scratch on 64x64 input",
              "  - MobileNetV2-A: frozen base, head-only training on 96x96 input",
              "  - MobileNetV2-FT: fine-tune last 10 layers on 96x96 -- FAILED (overfitting)",
              ""]
    path = OUTPUTS / "benchmark_summary.txt"
    path.write_text("\n".join(lines), encoding="utf-8")
    log.info("Saved: %s", path)
    return path


# ─────────────────────────────────────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────────────────────────────────────

def main():
    log.info("=" * 60)
    log.info("LeafIQ -- Phase 4: Benchmarking")
    log.info("=" * 60)

    rows = load_eval_results()

    log.info("\nGenerating comparison charts ...")
    plot_comparison_bars(rows)
    plot_params_vs_accuracy(rows)
    plot_inference_time(rows)
    plot_training_overlay(rows)

    log.info("\nSaving summary tables ...")
    save_benchmark_csv(rows)
    save_benchmark_txt(rows)

    log.info("\nBenchmark complete.")
    log.info("  Plots    : %s", PLOTS)
    log.info("  CSV      : %s", OUTPUTS / "benchmark_summary.csv")
    log.info("  Text     : %s", OUTPUTS / "benchmark_summary.txt")


if __name__ == "__main__":
    main()
