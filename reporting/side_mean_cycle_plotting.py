"""Graphique du cycle moyen du squat en vue latérale."""

from __future__ import annotations

import os

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np


def _plot_individual_cycles(axis, cycle_percent, cycles) -> None:
    for cycle in cycles:
        axis.plot(
            cycle_percent,
            cycle,
            color="gray",
            linewidth=0.9,
            alpha=0.25,
        )


def _plot_mean_band(
    axis,
    cycle_percent,
    mean_values,
    sd_values,
    color: str,
    label: str,
) -> None:
    axis.plot(
        cycle_percent,
        mean_values,
        color=color,
        linewidth=2.5,
        label=label,
    )
    axis.fill_between(
        cycle_percent,
        mean_values - sd_values,
        mean_values + sd_values,
        color=color,
        alpha=0.18,
        label="Moyenne ± 1 ET" if label == "Genou moyen" else None,
    )


def _style_axis(axis, ylabel: str) -> None:
    axis.set_ylabel(ylabel)
    axis.axhline(0, color="black", linestyle=":", linewidth=1)
    axis.grid(True, alpha=0.25)
    axis.legend(loc="upper left", fontsize=9)


def plot_side_mean_cycle(
    mean_cycle: dict | None,
    results_folder: str,
) -> str | None:
    """Trace les cycles individuels, la moyenne et l'écart-type."""

    if mean_cycle is None:
        return None

    cycle_percent = np.asarray(
        mean_cycle["cycle_percent"],
        dtype=float,
    )
    mean_bottom_percent = float(
        np.nanmean(mean_cycle["bottom_cycle_percent"])
    )

    figure, axes = plt.subplots(
        4,
        1,
        figsize=(12, 13),
        sharex=True,
        gridspec_kw={"height_ratios": [1.1, 1.1, 0.9, 0.9]},
    )
    knee_axis, hip_trunk_axis, velocity_axis, secondary_axis = axes

    for axis in axes:
        axis.axvline(
            mean_bottom_percent,
            color="black",
            linestyle="--",
            linewidth=1.2,
            alpha=0.7,
            label=(
                f"Point bas moyen ({mean_bottom_percent:.1f} %)"
                if axis is knee_axis
                else None
            ),
        )

    _plot_individual_cycles(
        knee_axis,
        cycle_percent,
        mean_cycle["knee_flexion_cycles"],
    )
    _plot_mean_band(
        knee_axis,
        cycle_percent,
        mean_cycle["knee_flexion_mean"],
        mean_cycle["knee_flexion_sd"],
        "tab:blue",
        "Genou moyen",
    )
    knee_axis.set_title(
        "Cycle moyen du squat — vue latérale "
        f"(n = {mean_cycle['n_cycles']})"
    )
    _style_axis(knee_axis, "Flexion genou (°)")

    _plot_mean_band(
        hip_trunk_axis,
        cycle_percent,
        mean_cycle["hip_flexion_mean"],
        mean_cycle["hip_flexion_sd"],
        "tab:orange",
        "Hanche moyenne",
    )
    _plot_mean_band(
        hip_trunk_axis,
        cycle_percent,
        mean_cycle["trunk_flexion_mean"],
        mean_cycle["trunk_flexion_sd"],
        "tab:green",
        "Tronc moyen",
    )
    _style_axis(hip_trunk_axis, "Angle relatif (°)")

    _plot_individual_cycles(
        velocity_axis,
        cycle_percent,
        mean_cycle["knee_velocity_cycles"],
    )
    _plot_mean_band(
        velocity_axis,
        cycle_percent,
        mean_cycle["knee_velocity_mean"],
        mean_cycle["knee_velocity_sd"],
        "tab:red",
        "Vitesse moyenne du genou",
    )
    velocity_axis.text(
        0.01,
        0.04,
        "Positive = descente | Négative = remontée",
        transform=velocity_axis.transAxes,
        fontsize=9,
    )
    _style_axis(velocity_axis, "Vitesse genou (°/s)")

    _plot_mean_band(
        secondary_axis,
        cycle_percent,
        mean_cycle["ankle_dorsiflexion_mean"],
        mean_cycle["ankle_dorsiflexion_sd"],
        "tab:purple",
        "Dorsiflexion moyenne",
    )
    _plot_mean_band(
        secondary_axis,
        cycle_percent,
        mean_cycle["foot_inclination_mean"],
        mean_cycle["foot_inclination_sd"],
        "tab:cyan",
        "Inclinaison moyenne du pied (expérimental)",
    )
    _style_axis(secondary_axis, "Mesures secondaires (°)")
    secondary_axis.set_xlabel("Cycle du squat (%)")
    secondary_axis.set_xlim(0.0, 100.0)

    figure.tight_layout()
    figure_path = os.path.join(
        results_folder,
        "sagittal_mean_cycle_plot.png",
    )
    figure.savefig(figure_path, dpi=300, bbox_inches="tight")
    plt.close(figure)

    return figure_path
