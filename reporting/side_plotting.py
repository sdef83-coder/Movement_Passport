"""Graphiques de contrôle pour l'analyse sagittale du squat."""

from __future__ import annotations

import os

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


REQUIRED_PLOT_COLUMNS = {
    "frame",
    "time_s",
    "knee_flexion_relative_deg",
    "knee_flexion_filtered_deg",
    "knee_angular_velocity_deg_s",
    "hip_flexion_filtered_deg",
    "trunk_flexion_filtered_deg",
    "ankle_dorsiflexion_filtered_deg",
    "foot_inclination_relative_deg",
    "foot_inclination_filtered_deg",
}


def _repetition_records(repetitions) -> list[dict]:
    if repetitions is None:
        return []

    if isinstance(repetitions, pd.DataFrame):
        return repetitions.to_dict(orient="records")

    return list(repetitions)


def _shade_repetitions(
    axes,
    repetitions: list[dict],
) -> None:
    for repetition in repetitions:
        for axis in axes:
            axis.axvspan(
                repetition["start_time_s"],
                repetition["end_time_s"],
                color="gray",
                alpha=0.10,
                linewidth=0,
            )


def _shade_invalid_zones(
    axes,
    dataframe: pd.DataFrame,
) -> None:
    if "signal_processing_valid" not in dataframe.columns:
        return

    invalid = ~dataframe["signal_processing_valid"].astype(bool).to_numpy()
    time_values = dataframe["time_s"].to_numpy(dtype=float)
    index = 0
    label_used = False

    while index < len(invalid):
        if not invalid[index]:
            index += 1
            continue

        start_index = index

        while index < len(invalid) and invalid[index]:
            index += 1

        end_index = min(index - 1, len(invalid) - 1)
        start_time = time_values[start_index]
        end_time = time_values[end_index]

        for axis in axes:
            axis.axvspan(
                start_time,
                end_time,
                color="red",
                alpha=0.08,
                linewidth=0,
                label=(
                    "Données invalides"
                    if not label_used and axis is axes[0]
                    else None
                ),
            )

        label_used = True


def _style_axis(axis, ylabel: str) -> None:
    axis.set_ylabel(ylabel)
    axis.axhline(
        0,
        color="black",
        linestyle=":",
        linewidth=1,
        alpha=0.7,
    )
    axis.grid(True, alpha=0.25)
    axis.legend(
        loc="upper left",
        frameon=True,
        fontsize=9,
    )


def plot_side_analysis(
    processed_dataframe: pd.DataFrame,
    repetitions,
    results_folder: str,
    heel_lift_threshold_deg: float = 20.0,
) -> str:
    """Trace les signaux sagittaux filtrés et la segmentation détectée."""

    missing_columns = sorted(
        REQUIRED_PLOT_COLUMNS - set(processed_dataframe.columns)
    )

    if missing_columns:
        raise ValueError(
            "Colonnes nécessaires au graphique sagittal manquantes : "
            + ", ".join(missing_columns)
        )

    if processed_dataframe.empty:
        raise ValueError("Le DataFrame sagittal traité est vide.")

    repetition_list = _repetition_records(repetitions)
    time_values = processed_dataframe["time_s"].to_numpy(dtype=float)

    figure, axes = plt.subplots(
        5,
        1,
        figsize=(13, 15.5),
        sharex=True,
        gridspec_kw={
            "height_ratios": [1.35, 0.85, 1.0, 1.0, 1.0],
        },
    )
    (
        knee_axis,
        knee_velocity_axis,
        hip_trunk_axis,
        ankle_axis,
        foot_axis,
    ) = axes

    _shade_repetitions(axes, repetition_list)
    _shade_invalid_zones(axes, processed_dataframe)

    # 1. Genou et segmentation
    knee_axis.plot(
        time_values,
        processed_dataframe["knee_flexion_relative_deg"],
        color="gray",
        alpha=0.35,
        linewidth=1,
        label="Genou relatif brut",
    )
    knee_axis.plot(
        time_values,
        processed_dataframe["knee_flexion_filtered_deg"],
        color="tab:blue",
        linewidth=2,
        label="Genou filtré",
    )

    for repetition in repetition_list:
        bottom_rows = processed_dataframe.loc[
            processed_dataframe["frame"] == repetition["bottom_frame"]
        ]

        if bottom_rows.empty:
            continue

        bottom_value = float(
            bottom_rows.iloc[0]["knee_flexion_filtered_deg"]
        )
        bottom_time = float(repetition["bottom_time_s"])
        knee_axis.scatter(
            bottom_time,
            bottom_value,
            color="red",
            s=32,
            zorder=5,
        )
        knee_axis.annotate(
            f"R{int(repetition['rep_id'])}",
            (bottom_time, bottom_value),
            xytext=(0, 8),
            textcoords="offset points",
            ha="center",
            fontsize=8,
        )

    knee_axis.set_title(
        "Analyse sagittale du squat — segmentation, angles et vitesse"
    )
    knee_axis.margins(y=0.12)
    _style_axis(knee_axis, "Flexion genou (°)")

    # 2. Vitesse angulaire continue du genou
    knee_velocity_axis.plot(
        time_values,
        processed_dataframe["knee_angular_velocity_deg_s"],
        color="tab:red",
        linewidth=1.8,
        label="Vitesse genou (+ descente / - remontée)",
    )
    _style_axis(knee_velocity_axis, "Vitesse genou (°/s)")

    # 3. Hanche et tronc
    hip_trunk_axis.plot(
        time_values,
        processed_dataframe["hip_flexion_filtered_deg"],
        color="tab:orange",
        linewidth=2,
        label="Hanche filtrée",
    )
    hip_trunk_axis.plot(
        time_values,
        processed_dataframe["trunk_flexion_filtered_deg"],
        color="tab:green",
        linewidth=2,
        label="Tronc filtré",
    )
    _style_axis(hip_trunk_axis, "Angle relatif (°)")

    # 4. Cheville
    ankle_axis.plot(
        time_values,
        processed_dataframe["ankle_dorsiflexion_filtered_deg"],
        color="tab:purple",
        linewidth=2,
        label="Dorsiflexion filtrée",
    )
    _style_axis(ankle_axis, "Dorsiflexion (°)")

    # 5. Inclinaison du pied
    foot_axis.plot(
        time_values,
        processed_dataframe["foot_inclination_relative_deg"],
        color="gray",
        alpha=0.30,
        linewidth=1,
        label="Pied relatif brut",
    )
    foot_axis.plot(
        time_values,
        processed_dataframe["foot_inclination_filtered_deg"],
        color="tab:cyan",
        linewidth=2,
        label="Pied filtré",
    )

    if np.isfinite(heel_lift_threshold_deg):
        foot_axis.axhline(
            heel_lift_threshold_deg,
            color="tab:red",
            linestyle="--",
            linewidth=1.5,
            label=(
                "Seuil expérimental de décollement "
                f"({heel_lift_threshold_deg:g}°)"
            ),
        )

    _style_axis(foot_axis, "Inclinaison pied (°)")
    foot_axis.set_xlabel("Temps (s)")

    figure.tight_layout()

    figure_path = os.path.join(
        results_folder,
        "sagittal_analysis_plot.png",
    )
    figure.savefig(
        figure_path,
        dpi=300,
        bbox_inches="tight",
    )
    plt.close(figure)

    return figure_path
