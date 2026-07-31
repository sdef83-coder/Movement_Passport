"""Rapport descriptif du squat en vue latérale."""

from __future__ import annotations

import numpy as np
import pandas as pd


def _format_value(value, decimals: int = 1, suffix: str = "") -> str:
    try:
        numeric_value = float(value)
    except (TypeError, ValueError):
        return "Non disponible"

    if not np.isfinite(numeric_value):
        return "Non disponible"

    return f"{numeric_value:.{decimals}f}{suffix}"


def _phase_label(phase: str) -> str:
    return {
        "descent": "descente",
        "bottom": "point bas",
        "ascent": "remontée",
        "not_detected": "non détectée",
    }.get(str(phase), str(phase))


def _side_label(side: str) -> str:
    return {
        "left": "gauche",
        "right": "droite",
    }.get(str(side), str(side))


def _mean_sd_text(values: pd.Series, suffix: str = "°") -> str:
    numeric_values = pd.to_numeric(values, errors="coerce").dropna()

    if numeric_values.empty:
        return "Non disponible"

    mean_value = float(numeric_values.mean())
    standard_deviation = float(numeric_values.std(ddof=0))

    return f"{mean_value:.1f}{suffix} ± {standard_deviation:.1f}{suffix}"


def build_side_report(
    repetition_metrics: pd.DataFrame,
    metadata: dict | None = None,
) -> str:
    """Génère un rapport sagittal descriptif sans interprétation clinique."""

    lines = [
        "RAPPORT DESCRIPTIF - ANALYSE SAGITTALE DU SQUAT",
        "=" * 52,
        "",
    ]

    if repetition_metrics.empty:
        lines.extend(
            [
                "Aucune répétition valide n'a pu être analysée.",
                "",
                "Ce rapport ne constitue pas un diagnostic médical.",
            ]
        )
        return "\n".join(lines) + "\n"

    lines.extend(
        [
            "1. INFORMATIONS GÉNÉRALES",
            "-" * 30,
            f"Nombre de répétitions analysées : {len(repetition_metrics)}",
        ]
    )

    if metadata is not None:
        lines.append(
            "Côté analysé : "
            + _side_label(metadata.get("analysis_side", "inconnu"))
        )
        lines.append(
            "Durée de l'enregistrement : "
            + _format_value(metadata.get("duration_s"), 2, " s")
        )
        lines.append(
            "Visibilité valide : "
            + _format_value(
                metadata.get("visibility_ok_percent"), 1, " %"
            )
        )

    lines.append(
        "Qualité moyenne des répétitions : "
        + _format_value(
            repetition_metrics["quality_percent"].mean(), 1, " %"
        )
    )
    lines.append("")

    lines.extend(
        [
            "2. MESURES PRINCIPALES PAR RÉPÉTITION",
            "-" * 42,
        ]
    )

    for _, repetition in repetition_metrics.iterrows():
        lines.append(f"Répétition {int(repetition['rep_id'])}")
        lines.append(
            "  Durée : "
            + _format_value(repetition["duration_s"], 2, " s")
            + " (descente "
            + _format_value(repetition["descent_duration_s"], 2, " s")
            + ", remontée "
            + _format_value(repetition["ascent_duration_s"], 2, " s")
            + ")"
        )
        lines.append(
            "  Vitesse moyenne du genou — descente : "
            + _format_value(
                repetition["mean_knee_descent_velocity_deg_s"],
                1,
                "°/s",
            )
            + ", remontée : "
            + _format_value(
                repetition["mean_knee_ascent_velocity_deg_s"],
                1,
                "°/s",
            )
        )
        lines.append(
            "  Point bas — genou : "
            + _format_value(
                repetition["knee_flexion_at_bottom_deg"], 1, "°"
            )
            + ", hanche : "
            + _format_value(
                repetition["hip_flexion_at_bottom_deg"], 1, "°"
            )
            + ", tronc : "
            + _format_value(
                repetition["trunk_flexion_at_bottom_deg"], 1, "°"
            )
        )
        lines.append(
            "  Pics — genou : "
            + _format_value(repetition["peak_knee_flexion_deg"], 1, "°")
            + ", hanche : "
            + _format_value(repetition["peak_hip_flexion_deg"], 1, "°")
            + ", tronc : "
            + _format_value(repetition["peak_trunk_flexion_deg"], 1, "°")
        )
        lines.append(
            "  Pic de hanche : "
            + _phase_label(repetition["peak_hip_flexion_phase"])
            + " à "
            + _format_value(
                repetition["peak_hip_flexion_cycle_percent"],
                1,
                " % du cycle",
            )
        )
        lines.append(
            "  Pic du tronc : "
            + _phase_label(repetition["peak_trunk_flexion_phase"])
            + " à "
            + _format_value(
                repetition["peak_trunk_flexion_cycle_percent"],
                1,
                " % du cycle",
            )
        )
        lines.append(
            "  Qualité : "
            + _format_value(repetition["quality_percent"], 1, " %")
            + "; données interpolées : "
            + _format_value(repetition["interpolated_percent"], 1, " %")
        )
        lines.append("")

    deepest_index = repetition_metrics["peak_knee_flexion_deg"].idxmax()
    deepest_repetition = repetition_metrics.loc[deepest_index]
    trunk_index = repetition_metrics["peak_trunk_flexion_deg"].idxmax()
    trunk_repetition = repetition_metrics.loc[trunk_index]

    lines.extend(
        [
            "3. SYNTHÈSE DESCRIPTIVE",
            "-" * 30,
            "Pic moyen de flexion du genou : "
            + _mean_sd_text(repetition_metrics["peak_knee_flexion_deg"]),
            "Pic moyen de flexion de hanche : "
            + _mean_sd_text(repetition_metrics["peak_hip_flexion_deg"]),
            "Pic moyen de flexion du tronc : "
            + _mean_sd_text(repetition_metrics["peak_trunk_flexion_deg"]),
            "Durée moyenne : "
            + _mean_sd_text(repetition_metrics["duration_s"], " s"),
            "Vitesse moyenne de descente du genou : "
            + _mean_sd_text(
                repetition_metrics["mean_knee_descent_velocity_deg_s"],
                "°/s",
            ),
            "Vitesse moyenne de remontée du genou : "
            + _mean_sd_text(
                repetition_metrics["mean_knee_ascent_velocity_deg_s"],
                "°/s",
            ),
            "Plus grande flexion du genou : répétition "
            f"{int(deepest_repetition['rep_id'])} "
            + "("
            + _format_value(
                deepest_repetition["peak_knee_flexion_deg"], 1, "°"
            )
            + ").",
            "Plus grande flexion du tronc : répétition "
            f"{int(trunk_repetition['rep_id'])} "
            + "("
            + _format_value(
                trunk_repetition["peak_trunk_flexion_deg"], 1, "°"
            )
            + ").",
            "",
        ]
    )

    above_threshold_count = int(
        repetition_metrics["heel_lift_screening"]
        .eq("above_experimental_threshold")
        .sum()
    )
    threshold_value = repetition_metrics["heel_lift_threshold_deg"].iloc[0]

    lines.extend(
        [
            "4. MESURES SECONDAIRES ET EXPÉRIMENTALES",
            "-" * 46,
            "Pic moyen de dorsiflexion relative : "
            + _mean_sd_text(
                repetition_metrics["peak_ankle_dorsiflexion_deg"]
            ),
            "Pic moyen d'inclinaison relative du pied : "
            + _mean_sd_text(
                repetition_metrics["peak_foot_inclination_deg"]
            ),
            "Répétitions au-dessus du seuil expérimental du pied ("
            + _format_value(threshold_value, 1, "°")
            + f") : {above_threshold_count}/{len(repetition_metrics)}.",
            "Ces mesures sont fournies à titre exploratoire et ne doivent pas "
            "être interprétées comme une confirmation clinique.",
            "",
            "5. LIMITES",
            "-" * 30,
            "Les angles sont issus d'une estimation vidéo 2D et sont corrigés "
            "par rapport à la posture de baseline.",
            "Ils permettent surtout des comparaisons entre répétitions "
            "réalisées dans les mêmes conditions.",
            "La dorsiflexion et l'inclinaison du pied dépendent fortement de la "
            "qualité des points du talon et de l'avant-pied.",
            "Ce rapport est descriptif et ne constitue pas un diagnostic "
            "médical.",
        ]
    )

    return "\n".join(lines) + "\n"
