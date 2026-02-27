"""
Dashboard de suivi d'entraînement YOLO26 (et YOLOv11).

Lit le fichier results.csv généré automatiquement par Ultralytics pendant
l'entraînement et affiche en temps réel :
  - Losses (box, cls, dfl) train + val
  - Métriques (mAP50, mAP50-95, precision, recall)
  - Learning rate
  - Temps par époque

Deux modes :
  1. Live (défaut) : rafraîchit toutes les 10s pendant l'entraînement
  2. Static : affiche les courbes finales d'un run terminé

Usage :
  # Suivre un entraînement en cours (auto-détecte le dernier run)
  python training/dashboard.py --live

  # Suivre un run spécifique
  python training/dashboard.py --run training/runs_yolo26/yolo26_ball_20260225_143000

  # Afficher les résultats d'un run terminé (pas de refresh)
  python training/dashboard.py --run training/runs_yolo26/yolo26_ball_20260225_143000 --static

  # Comparer plusieurs runs
  python training/dashboard.py --compare run1_path run2_path run3_path

  # Sauvegarder en PNG sans afficher
  python training/dashboard.py --run path/to/run --save dashboard.png
"""

import argparse
import sys
import time
from pathlib import Path

import matplotlib
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import numpy as np
import pandas as pd


# ---------------------------------------------------------------------------
# Parsing du results.csv Ultralytics
# ---------------------------------------------------------------------------

# Colonnes standard du results.csv Ultralytics (YOLO11 / YOLO26)
# Note : YOLO26 peut ne pas avoir dfl_loss (DFL supprimé), le dashboard gère
EXPECTED_COLUMNS = [
    "epoch",
    "train/box_loss",
    "train/cls_loss",
    "train/dfl_loss",
    "metrics/precision(B)",
    "metrics/recall(B)",
    "metrics/mAP50(B)",
    "metrics/mAP50-95(B)",
    "val/box_loss",
    "val/cls_loss",
    "val/dfl_loss",
    "lr/pg0",
    "lr/pg1",
    "lr/pg2",
]


def load_results(results_csv: Path) -> pd.DataFrame:
    """Charge et nettoie le results.csv d'Ultralytics."""
    if not results_csv.exists():
        raise FileNotFoundError(f"Fichier non trouvé : {results_csv}")

    df = pd.read_csv(results_csv, skipinitialspace=True)
    # Nettoyer les noms de colonnes (espaces en trop)
    df.columns = df.columns.str.strip()
    return df


def find_latest_run(search_dirs: list[Path]) -> Path | None:
    """Trouve le run le plus récent dans les dossiers de runs."""
    latest = None
    latest_time = 0

    for runs_dir in search_dirs:
        if not runs_dir.exists():
            continue
        for run_dir in runs_dir.iterdir():
            if not run_dir.is_dir():
                continue
            csv = run_dir / "results.csv"
            if csv.exists() and csv.stat().st_mtime > latest_time:
                latest_time = csv.stat().st_mtime
                latest = run_dir

    return latest


# ---------------------------------------------------------------------------
# Plotting
# ---------------------------------------------------------------------------

COLORS = {
    "train": "#2196F3",      # bleu
    "val": "#FF5722",        # orange-rouge
    "precision": "#4CAF50",  # vert
    "recall": "#FF9800",     # orange
    "mAP50": "#9C27B0",      # violet
    "mAP50-95": "#E91E63",   # rose
    "lr": "#607D8B",         # gris
}


def plot_dashboard(df: pd.DataFrame, run_name: str = "", save_path: str = None):
    """
    Affiche le dashboard complet pour un run.

    Layout (2x3) :
      [Train Losses] [Val Losses]   [mAP]
      [Prec/Recall]  [LR Schedule]  [Summary]
    """
    fig = plt.figure(figsize=(18, 10))
    fig.suptitle(
        f"YOLO Training Dashboard — {run_name}" if run_name else "YOLO Training Dashboard",
        fontsize=14,
        fontweight="bold",
        y=0.98,
    )

    gs = gridspec.GridSpec(2, 3, hspace=0.35, wspace=0.3, top=0.93, bottom=0.07)
    epochs = df["epoch"] if "epoch" in df.columns else np.arange(len(df))

    # ---- 1. Train Losses --------------------------------------------------
    ax1 = fig.add_subplot(gs[0, 0])
    ax1.set_title("Train Losses", fontweight="bold")

    if "train/box_loss" in df.columns:
        ax1.plot(epochs, df["train/box_loss"], label="box_loss", color="#2196F3", linewidth=1.5)
    if "train/cls_loss" in df.columns:
        ax1.plot(epochs, df["train/cls_loss"], label="cls_loss", color="#FF9800", linewidth=1.5)
    if "train/dfl_loss" in df.columns:
        vals = df["train/dfl_loss"]
        if not vals.isna().all() and (vals != 0).any():
            ax1.plot(epochs, vals, label="dfl_loss", color="#4CAF50", linewidth=1.5)

    ax1.set_xlabel("Epoch")
    ax1.set_ylabel("Loss")
    ax1.legend(loc="upper right", fontsize=8)
    ax1.grid(True, alpha=0.3)

    # ---- 2. Val Losses ----------------------------------------------------
    ax2 = fig.add_subplot(gs[0, 1])
    ax2.set_title("Validation Losses", fontweight="bold")

    if "val/box_loss" in df.columns:
        ax2.plot(epochs, df["val/box_loss"], label="box_loss", color="#2196F3", linewidth=1.5)
    if "val/cls_loss" in df.columns:
        ax2.plot(epochs, df["val/cls_loss"], label="cls_loss", color="#FF9800", linewidth=1.5)
    if "val/dfl_loss" in df.columns:
        vals = df["val/dfl_loss"]
        if not vals.isna().all() and (vals != 0).any():
            ax2.plot(epochs, vals, label="dfl_loss", color="#4CAF50", linewidth=1.5)

    ax2.set_xlabel("Epoch")
    ax2.set_ylabel("Loss")
    ax2.legend(loc="upper right", fontsize=8)
    ax2.grid(True, alpha=0.3)

    # ---- 3. mAP -----------------------------------------------------------
    ax3 = fig.add_subplot(gs[0, 2])
    ax3.set_title("Mean Average Precision", fontweight="bold")

    if "metrics/mAP50(B)" in df.columns:
        ax3.plot(epochs, df["metrics/mAP50(B)"], label="mAP@50", color="#9C27B0", linewidth=2)
    if "metrics/mAP50-95(B)" in df.columns:
        ax3.plot(epochs, df["metrics/mAP50-95(B)"], label="mAP@50-95", color="#E91E63", linewidth=2)

    ax3.set_xlabel("Epoch")
    ax3.set_ylabel("mAP")
    ax3.set_ylim(0, 1.0)
    ax3.legend(loc="lower right", fontsize=9)
    ax3.grid(True, alpha=0.3)

    # Annoter le meilleur mAP50
    if "metrics/mAP50(B)" in df.columns and len(df) > 0:
        best_idx = df["metrics/mAP50(B)"].idxmax()
        best_val = df["metrics/mAP50(B)"].iloc[best_idx]
        best_epoch = epochs.iloc[best_idx] if hasattr(epochs, "iloc") else best_idx
        ax3.annotate(
            f"Best: {best_val:.3f} (ep {int(best_epoch)})",
            xy=(best_epoch, best_val),
            xytext=(best_epoch, best_val - 0.08),
            fontsize=8,
            fontweight="bold",
            color="#9C27B0",
            arrowprops=dict(arrowstyle="->", color="#9C27B0", lw=1.2),
        )

    # ---- 4. Precision / Recall --------------------------------------------
    ax4 = fig.add_subplot(gs[1, 0])
    ax4.set_title("Precision & Recall", fontweight="bold")

    if "metrics/precision(B)" in df.columns:
        ax4.plot(epochs, df["metrics/precision(B)"], label="Precision", color="#4CAF50", linewidth=1.5)
    if "metrics/recall(B)" in df.columns:
        ax4.plot(epochs, df["metrics/recall(B)"], label="Recall", color="#FF9800", linewidth=1.5)

    ax4.set_xlabel("Epoch")
    ax4.set_ylabel("Score")
    ax4.set_ylim(0, 1.0)
    ax4.legend(loc="lower right", fontsize=9)
    ax4.grid(True, alpha=0.3)

    # ---- 5. Learning Rate -------------------------------------------------
    ax5 = fig.add_subplot(gs[1, 1])
    ax5.set_title("Learning Rate Schedule", fontweight="bold")

    lr_cols = [c for c in df.columns if c.startswith("lr/")]
    for i, col in enumerate(lr_cols):
        label = col.replace("lr/", "")
        ax5.plot(epochs, df[col], label=label, linewidth=1.2, alpha=0.8)

    ax5.set_xlabel("Epoch")
    ax5.set_ylabel("Learning Rate")
    ax5.legend(loc="upper right", fontsize=8)
    ax5.grid(True, alpha=0.3)
    ax5.ticklabel_format(style="scientific", axis="y", scilimits=(0, 0))

    # ---- 6. Summary Table -------------------------------------------------
    ax6 = fig.add_subplot(gs[1, 2])
    ax6.axis("off")
    ax6.set_title("Summary", fontweight="bold")

    if len(df) > 0:
        last = df.iloc[-1]
        total_epochs = int(last["epoch"]) + 1 if "epoch" in df.columns else len(df)

        summary_data = [
            ["Epochs", f"{total_epochs}"],
        ]

        # Meilleurs scores
        if "metrics/mAP50(B)" in df.columns:
            best_map50 = df["metrics/mAP50(B)"].max()
            best_ep = int(df["metrics/mAP50(B)"].idxmax())
            summary_data.append(["Best mAP@50", f"{best_map50:.4f} (ep {best_ep})"])

        if "metrics/mAP50-95(B)" in df.columns:
            best_map = df["metrics/mAP50-95(B)"].max()
            summary_data.append(["Best mAP@50-95", f"{best_map:.4f}"])

        if "metrics/precision(B)" in df.columns:
            best_p = df["metrics/precision(B)"].max()
            summary_data.append(["Best Precision", f"{best_p:.4f}"])

        if "metrics/recall(B)" in df.columns:
            best_r = df["metrics/recall(B)"].max()
            summary_data.append(["Best Recall", f"{best_r:.4f}"])

        # Dernières losses
        if "train/box_loss" in df.columns:
            summary_data.append(["Last box_loss", f"{last['train/box_loss']:.4f}"])
        if "train/cls_loss" in df.columns:
            summary_data.append(["Last cls_loss", f"{last['train/cls_loss']:.4f}"])

        # LR actuel
        if lr_cols:
            summary_data.append(["Current LR", f"{last[lr_cols[0]]:.6f}"])

        table = ax6.table(
            cellText=summary_data,
            colLabels=["Metric", "Value"],
            loc="center",
            cellLoc="left",
            colWidths=[0.45, 0.45],
        )
        table.auto_set_font_size(False)
        table.set_fontsize(9)
        table.scale(1, 1.6)

        # Style du header
        for (row, col), cell in table.get_celld().items():
            if row == 0:
                cell.set_facecolor("#37474F")
                cell.set_text_props(color="white", fontweight="bold")
            elif row % 2 == 0:
                cell.set_facecolor("#F5F5F5")
            cell.set_edgecolor("#E0E0E0")

    if save_path:
        fig.savefig(save_path, dpi=150, bbox_inches="tight", facecolor="white")
        print(f"Dashboard sauvegardé : {save_path}")

    return fig


def plot_comparison(run_paths: list[Path], save_path: str = None):
    """Compare les courbes de plusieurs runs sur le meme graphique."""
    fig, axes = plt.subplots(1, 3, figsize=(18, 5))
    fig.suptitle("Comparaison des runs", fontsize=14, fontweight="bold")

    cmap = plt.cm.tab10
    metrics_to_plot = [
        ("metrics/mAP50(B)", "mAP@50", axes[0]),
        ("metrics/mAP50-95(B)", "mAP@50-95", axes[1]),
        ("train/box_loss", "Train Box Loss", axes[2]),
    ]

    for i, run_path in enumerate(run_paths):
        csv_path = Path(run_path) / "results.csv"
        if not csv_path.exists():
            print(f"Pas de results.csv dans {run_path}, ignoré")
            continue

        df = load_results(csv_path)
        run_name = Path(run_path).name
        color = cmap(i % 10)
        epochs = df["epoch"] if "epoch" in df.columns else np.arange(len(df))

        for col, title, ax in metrics_to_plot:
            if col in df.columns:
                ax.plot(epochs, df[col], label=run_name, color=color, linewidth=1.5)

    for col, title, ax in metrics_to_plot:
        ax.set_title(title, fontweight="bold")
        ax.set_xlabel("Epoch")
        ax.legend(fontsize=7, loc="best")
        ax.grid(True, alpha=0.3)

    plt.tight_layout()

    if save_path:
        fig.savefig(save_path, dpi=150, bbox_inches="tight", facecolor="white")
        print(f"Comparaison sauvegardée : {save_path}")

    return fig


# ---------------------------------------------------------------------------
# Mode Live
# ---------------------------------------------------------------------------

def live_dashboard(run_dir: Path, interval: int = 10):
    """
    Rafraîchit le dashboard toutes les `interval` secondes.
    Ctrl+C pour arrêter.
    """
    plt.ion()
    fig = None
    csv_path = run_dir / "results.csv"

    print(f"Mode live - surveillance de : {csv_path}")
    print(f"Rafraîchissement toutes les {interval}s. Ctrl+C pour quitter.\n")

    try:
        while True:
            if csv_path.exists():
                try:
                    df = load_results(csv_path)
                    if len(df) > 0:
                        if fig is not None:
                            plt.close(fig)
                        fig = plot_dashboard(df, run_name=run_dir.name)
                        plt.pause(0.1)
                        print(
                            f"\r  Epoch {len(df)} | "
                            f"mAP50: {df['metrics/mAP50(B)'].iloc[-1]:.3f} | "
                            f"box_loss: {df['train/box_loss'].iloc[-1]:.4f}  ",
                            end="",
                            flush=True,
                        )
                except Exception as e:
                    print(f"\rErreur lecture CSV : {e}  ", end="", flush=True)
            else:
                print(f"\rEn attente de {csv_path}...  ", end="", flush=True)

            time.sleep(interval)

    except KeyboardInterrupt:
        print("\n\nDashboard live arrêté.")
    finally:
        plt.ioff()
        if fig is not None:
            plt.close(fig)


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main():
    training_dir = Path(__file__).parent

    parser = argparse.ArgumentParser(
        description="Dashboard de suivi d'entraînement YOLO",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        "--run",
        type=str,
        default=None,
        help="Chemin vers un dossier de run (contenant results.csv)",
    )
    parser.add_argument(
        "--live",
        action="store_true",
        help="Mode live : rafraîchit pendant l'entraînement",
    )
    parser.add_argument(
        "--static",
        action="store_true",
        help="Mode statique : affiche une seule fois",
    )
    parser.add_argument(
        "--interval",
        type=int,
        default=10,
        help="Intervalle de rafraîchissement en secondes (mode live)",
    )
    parser.add_argument(
        "--compare",
        nargs="+",
        type=str,
        default=None,
        help="Comparer plusieurs runs (chemins vers les dossiers)",
    )
    parser.add_argument(
        "--save",
        type=str,
        default=None,
        help="Sauvegarder le dashboard en image (PNG/PDF)",
    )

    args = parser.parse_args()

    # -- Mode comparaison ---------------------------------------------------
    if args.compare:
        fig = plot_comparison([Path(p) for p in args.compare], save_path=args.save)
        if not args.save:
            plt.show()
        return

    # -- Trouver le run -----------------------------------------------------
    if args.run:
        run_dir = Path(args.run)
    else:
        # Auto-detect le dernier run
        search_dirs = [
            training_dir / "runs_yolo26",
            training_dir / "runs_combined",
            training_dir / "runs",
        ]
        run_dir = find_latest_run(search_dirs)
        if run_dir is None:
            print("Aucun run trouvé. Lancez un entraînement ou spécifiez --run.")
            print(f"Dossiers cherchés : {[str(d) for d in search_dirs]}")
            sys.exit(1)

    print(f"Run sélectionné : {run_dir}")

    csv_path = run_dir / "results.csv"

    # -- Mode live ----------------------------------------------------------
    if args.live or (not args.static and not args.save):
        # Par défaut = live si pas d'autre option
        if csv_path.exists():
            # Si le CSV existe et --static pas demandé, vérifier si l'entraînement
            # est en cours (le fichier a été modifié dans les dernières 60s)
            mtime = csv_path.stat().st_mtime
            if time.time() - mtime < 60 or args.live:
                live_dashboard(run_dir, interval=args.interval)
                return

        if not csv_path.exists() and args.live:
            # Attendre que le CSV apparaisse
            live_dashboard(run_dir, interval=args.interval)
            return

    # -- Mode statique ------------------------------------------------------
    if not csv_path.exists():
        print(f"Fichier non trouvé : {csv_path}")
        sys.exit(1)

    df = load_results(csv_path)
    if len(df) == 0:
        print("Le fichier results.csv est vide.")
        sys.exit(1)

    print(f"Chargement de {len(df)} époques depuis {csv_path}")

    fig = plot_dashboard(df, run_name=run_dir.name, save_path=args.save)

    if not args.save:
        plt.show()


if __name__ == "__main__":
    main()
