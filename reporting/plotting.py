import os
import matplotlib.pyplot as plt


def plot_movement_analysis(
    df,
    results_folder,
    filtered_pelvis_y=None,
    baseline_values=None,
    repetitions=None,
):
    """
    Affiche :
    - la hauteur du bassin et la segmentation ;
    - la déviation frontale signée des genoux ;
    - la vitesse verticale du pelvis.
    """

    fig, (ax_pelvis, ax_deviation, ax_velocity) = plt.subplots(
        3,
        1,
        figsize=(11, 10),
        sharex=True,
        gridspec_kw={"height_ratios": [2, 1, 1]},
    )

    # ========================================================
    # 1. Pelvis et segmentation
    # ========================================================
    ax_pelvis_y = ax_pelvis.twinx()

    ax_pelvis_y.plot(
        df["time_s"],
        df["pelvis_y"],
        label="Pelvis y brut",
        linestyle="--",
        color="black",
        alpha=0.4,
    )

    if filtered_pelvis_y is not None:
        ax_pelvis_y.plot(
            df["time_s"],
            filtered_pelvis_y,
            label="Pelvis y filtré",
            color="black",
            linewidth=2,
        )

    if baseline_values is not None:
        pelvis_baseline = baseline_values["pelvis_y_mean"]
        pelvis_std = baseline_values.get("pelvis_y_std", 0)

        start_threshold = pelvis_baseline + 3 * pelvis_std
        end_threshold = pelvis_baseline + 2 * pelvis_std

        ax_pelvis_y.axhline(
            pelvis_baseline,
            linestyle=":",
            linewidth=2,
            color="blue",
            label="Baseline pelvis",
        )

        ax_pelvis_y.axhline(
            start_threshold,
            linestyle=":",
            linewidth=2,
            color="orange",
            label="Seuil début",
        )

        ax_pelvis_y.axhline(
            end_threshold,
            linestyle=":",
            linewidth=2,
            color="green",
            label="Seuil fin",
        )

    if repetitions is not None:
        for rep in repetitions:
            ax_pelvis.axvspan(
                rep["start_time_s"],
                rep["end_time_s"],
                alpha=0.12,
                color="gray",
            )

            ax_pelvis.axvline(
                rep["start_time_s"],
                linestyle="--",
                linewidth=1,
                color="green",
            )

            ax_pelvis.axvline(
                rep["end_time_s"],
                linestyle="--",
                linewidth=1,
                color="red",
            )

            bottom_rows = df.loc[
                df["frame"] == rep["bottom_frame"],
                "pelvis_y",
            ]

            if not bottom_rows.empty:
                ax_pelvis_y.scatter(
                    rep["bottom_time_s"],
                    bottom_rows.iloc[0],
                    color="red",
                    zorder=5,
                )

    ax_pelvis.set_ylabel("Répétitions")
    ax_pelvis_y.set_ylabel("Pelvis y")
    ax_pelvis_y.invert_yaxis()
    ax_pelvis.grid(True)
    ax_pelvis.set_title(
        "Segmentation du mouvement, déviation frontale et vitesse"
    )

    lines, labels = ax_pelvis_y.get_legend_handles_labels()
    ax_pelvis_y.legend(lines, labels, loc="best")

    # ========================================================
    # 2. Déviation frontale signée
    # ========================================================
    ax_deviation.plot(
        df["time_s"],
        df["left_signed_deviation_percent"],
        label="Déviation gauche",
    )

    ax_deviation.plot(
        df["time_s"],
        df["right_signed_deviation_percent"],
        label="Déviation droite",
    )

    ax_deviation.axhline(
        0,
        linestyle=":",
        linewidth=1.5,
        color="black",
        label="Alignement neutre",
    )

    ax_deviation.axhspan(
        -2,
        2,
        alpha=0.10,
        color="gray",
        label="Zone neutre provisoire",
    )

    if repetitions is not None:
        for rep in repetitions:
            ax_deviation.axvspan(
                rep["start_time_s"],
                rep["end_time_s"],
                alpha=0.10,
                color="gray",
            )

    ax_deviation.set_ylabel("Déviation signée (%)")
    ax_deviation.grid(True)
    ax_deviation.legend()

    ax_deviation.text(
        0.01,
        0.04,
        "Négatif = valgus | Positif = varus",
        transform=ax_deviation.transAxes,
        fontsize=9,
    )

    # ========================================================
    # 3. Vitesse verticale du pelvis
    # ========================================================
    if "pelvis_velocity" in df.columns:
        ax_velocity.plot(
            df["time_s"],
            df["pelvis_velocity"],
            label="Vitesse verticale du pelvis",
            linewidth=2,
        )

        ax_velocity.axhline(
            0,
            color="black",
            linestyle=":",
            linewidth=1.5,
        )

        if repetitions is not None:
            for rep in repetitions:
                ax_velocity.axvspan(
                    rep["start_time_s"],
                    rep["end_time_s"],
                    alpha=0.10,
                    color="gray",
                )

        ax_velocity.fill_between(
            df["time_s"],
            0,
            df["pelvis_velocity"],
            where=df["pelvis_velocity"] > 0,
            alpha=0.2,
        )

        ax_velocity.fill_between(
            df["time_s"],
            0,
            df["pelvis_velocity"],
            where=df["pelvis_velocity"] < 0,
            alpha=0.2,
        )

    ax_velocity.set_ylabel("Vitesse\n(unités norm./s)")
    ax_velocity.set_xlabel("Temps (s)")
    ax_velocity.grid(True)
    ax_velocity.legend()

    ax_velocity.text(
        0.01,
        0.04,
        "Positive = descente | Négative = remontée",
        transform=ax_velocity.transAxes,
        fontsize=9,
    )

    figure_path = os.path.join(
        results_folder,
        "movement_analysis_plot.png",
    )

    fig.tight_layout()
    fig.savefig(
        figure_path,
        dpi=300,
        bbox_inches="tight",
    )
    plt.close(fig)

    return figure_path


def plot_fppa(
    df,
    results_folder,
    repetitions=None,
):
    """
    Affiche uniquement le FPPA gauche et droit.
    """

    fig, ax = plt.subplots(figsize=(11, 5))

    ax.plot(
        df["time_s"],
        df["left_fppa"],
        label="FPPA gauche",
    )

    ax.plot(
        df["time_s"],
        df["right_fppa"],
        label="FPPA droit",
    )

    if repetitions is not None:
        for rep in repetitions:
            ax.axvspan(
                rep["start_time_s"],
                rep["end_time_s"],
                alpha=0.12,
                color="gray",
            )

            ax.axvline(
                rep["bottom_time_s"],
                linestyle=":",
                linewidth=1,
                color="black",
            )

    ax.set_xlabel("Temps (s)")
    ax.set_ylabel("FPPA (degrés)")
    ax.set_title("Évolution du FPPA")
    ax.grid(True)
    ax.legend()

    figure_path = os.path.join(
        results_folder,
        "fppa_plot.png",
    )

    fig.tight_layout()
    fig.savefig(figure_path, dpi=300, bbox_inches="tight")
    plt.close(fig)

    return figure_path
