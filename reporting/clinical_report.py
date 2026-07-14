import numpy as np


def _format_value(value, decimals=2, suffix=""):
    """
    Formate proprement une valeur numérique.
    """
    if value is None or not np.isfinite(value):
        return "Non disponible"

    return f"{value:.{decimals}f}{suffix}"


def _format_direction(direction):
    """
    Traduit les états internes en français.
    """
    mapping = {
        "valgus": "valgus",
        "varus": "varus",
        "neutral": "neutre",
        "not_detected": "non détecté",
        "descent": "descente",
        "ascent": "remontée",
        "bottom": "point bas",
        "left": "gauche",
        "right": "droite",
        "equal": "équivalent",
        "simultaneous": "simultané",
        "bilateral_valgus": "valgus bilatéral",
        "bilateral_varus": "varus bilatéral",
        "left_valgus_right_varus": "valgus gauche et varus droit",
        "left_varus_right_valgus": "varus gauche et valgus droit",
        "neutral_or_mixed": "neutre ou mixte",
    }

    return mapping.get(str(direction), str(direction))


def build_clinical_report(
    clinical_df,
    metadata=None,
    mean_cycle_summary=None,
):
    """
    Génère un rapport clinique descriptif en texte
    à partir du DataFrame clinique des répétitions.

    Le rapport ne constitue pas un diagnostic médical.
    """

    if clinical_df.empty:
        return (
            "RAPPORT CLINIQUE - ANALYSE FRONTALE DU SQUAT\n"
            "================================================\n\n"
            "Aucune répétition valide n'a pu être analysée.\n"
        )

    lines = []

    lines.append("RAPPORT CLINIQUE - ANALYSE FRONTALE DU SQUAT")
    lines.append("=" * 52)
    lines.append("")

    # --------------------------------------------------------
    # 1. Informations générales
    # --------------------------------------------------------
    lines.append("1. INFORMATIONS GÉNÉRALES")
    lines.append("-" * 30)

    lines.append(f"Nombre de répétitions analysées : {len(clinical_df)}")

    if metadata is not None:
        duration = metadata.get("duration_s")
        visibility = metadata.get("visibility_ok_percent")

        if duration is not None:
            lines.append(f"Durée de l'enregistrement : {duration:.2f} s")

        if visibility is not None:
            lines.append(
                f"Visibilité des marqueurs valide : {visibility:.1f} %"
            )

    mean_quality = clinical_df["quality_percent"].mean()

    lines.append("Qualité moyenne des répétitions : " f"{mean_quality:.1f} %")
    lines.append("")

    # --------------------------------------------------------
    # 2. Résumé par répétition
    # --------------------------------------------------------
    lines.append("2. RÉSUMÉ PAR RÉPÉTITION")
    lines.append("-" * 30)

    for _, rep in clinical_df.iterrows():
        rep_id = int(rep["repetition"])

        lines.append(f"Répétition {rep_id}")

        lines.append(
            "  Durée totale : " + _format_value(rep["duration_s"], 2, " s")
        )

        lines.append(
            "  Descente : "
            + _format_value(
                rep["descent_duration_s"],
                2,
                " s",
            )
        )

        lines.append(
            "  Remontée : "
            + _format_value(
                rep["ascent_duration_s"],
                2,
                " s",
            )
        )

        lines.append(
            "  Gauche au point bas : "
            f"{_format_direction(rep['left_alignment_at_bottom'])}, "
            f"{_format_value(rep['left_deviation_at_bottom_percent'], 2, ' %')}"
        )

        lines.append(
            "  Droite au point bas : "
            f"{_format_direction(rep['right_alignment_at_bottom'])}, "
            f"{_format_value(rep['right_deviation_at_bottom_percent'], 2, ' %')}"
        )

        lines.append(
            "  Déviation maximale gauche : "
            f"{_format_value(rep['left_peak_deviation_percent'], 2, ' %')} "
            f"({_format_direction(rep['left_peak_deviation_direction'])}, "
            f"phase de {_format_direction(rep['left_peak_deviation_phase'])}, "
            f"{_format_value(rep['left_peak_deviation_cycle_percent'], 1, ' % du cycle')})"
        )

        lines.append(
            "  Déviation maximale droite : "
            f"{_format_value(rep['right_peak_deviation_percent'], 2, ' %')} "
            f"({_format_direction(rep['right_peak_deviation_direction'])}, "
            f"phase de {_format_direction(rep['right_peak_deviation_phase'])}, "
            f"{_format_value(rep['right_peak_deviation_cycle_percent'], 1, ' % du cycle')})"
        )

        lines.append(
            "  Vitesse moyenne de descente : "
            + _format_value(
                rep["mean_descent_velocity"],
                3,
                " unités norm./s",
            )
        )

        lines.append(
            "  Vitesse moyenne de remontée : "
            + _format_value(
                rep["mean_ascent_velocity"],
                3,
                " unités norm./s",
            )
        )

        lines.append(
            "  Vitesse maximale de descente : "
            + _format_value(
                rep["peak_descent_velocity"],
                3,
                " unités norm./s",
            )
        )

        lines.append(
            "  Vitesse maximale de remontée : "
            + _format_value(
                rep["peak_ascent_velocity"],
                3,
                " unités norm./s",
            )
        )
        lines.append(
            "  Asymétrie d'amplitude G/D : "
            + _format_value(
                rep["frontal_amplitude_difference_percent"],
                2,
                " points",
            )
        )

        lines.append(
            "  Côté présentant la plus grande déviation : "
            f"{_format_direction(rep['greater_deviation_side'])}"
        )

        lines.append(
            "  Décalage entre les pics gauche et droit : "
            + _format_value(
                rep["frontal_peak_timing_difference_cycle_percent"],
                1,
                " % du cycle",
            )
        )

        lines.append(
            "  Pic le plus précoce : "
            f"{_format_direction(rep['earlier_peak_side'])}"
        )

        lines.append(
            "  Pattern bilatéral (au pic de déviation) : "
            f"{_format_direction(rep['bilateral_peak_pattern'])}"
        )

        lines.append("")

    # --------------------------------------------------------
    # 3. Répétitions les plus défavorables
    # --------------------------------------------------------
    lines.append("3. RÉPÉTITIONS LES PLUS DÉFAVORABLES")
    lines.append("-" * 38)

    left_worst_idx = clinical_df["left_peak_deviation_percent"].idxmax()

    right_worst_idx = clinical_df["right_peak_deviation_percent"].idxmax()

    left_worst = clinical_df.loc[left_worst_idx]
    right_worst = clinical_df.loc[right_worst_idx]

    lines.append(
        "Côté gauche : répétition "
        f"{int(left_worst['repetition'])}, "
        f"{_format_value(left_worst['left_peak_deviation_percent'], 2, ' %')} "
        f"en {_format_direction(left_worst['left_peak_deviation_direction'])}, "
        f"pendant la {_format_direction(left_worst['left_peak_deviation_phase'])}."
    )

    lines.append(
        "Côté droit : répétition "
        f"{int(right_worst['repetition'])}, "
        f"{_format_value(right_worst['right_peak_deviation_percent'], 2, ' %')} "
        f"en {_format_direction(right_worst['right_peak_deviation_direction'])}, "
        f"pendant la {_format_direction(right_worst['right_peak_deviation_phase'])}."
    )

    lines.append("")

    # --------------------------------------------------------
    # 4. Synthèse descriptive globale
    # --------------------------------------------------------
    lines.append("4. SYNTHÈSE DESCRIPTIVE")
    lines.append("-" * 30)

    left_directions = clinical_df[
        "left_peak_deviation_direction"
    ].value_counts()

    right_directions = clinical_df[
        "right_peak_deviation_direction"
    ].value_counts()

    left_main_direction = left_directions.idxmax()
    right_main_direction = right_directions.idxmax()

    left_mean_peak = clinical_df["left_peak_deviation_percent"].mean()

    right_mean_peak = clinical_df["right_peak_deviation_percent"].mean()

    lines.append(
        "À gauche, la direction de déviation la plus fréquente "
        f"est le {_format_direction(left_main_direction)}, "
        f"avec une amplitude maximale moyenne de "
        f"{left_mean_peak:.2f} %."
    )

    lines.append(
        "À droite, la direction de déviation la plus fréquente "
        f"est le {_format_direction(right_main_direction)}, "
        f"avec une amplitude maximale moyenne de "
        f"{right_mean_peak:.2f} %."
    )

    mean_descent = clinical_df["mean_descent_velocity"].mean()

    mean_ascent = clinical_df["mean_ascent_velocity"].mean()

    lines.append(
        "La vitesse moyenne de descente sur l'ensemble des "
        f"répétitions est de {mean_descent:.3f} unités norm./s."
    )

    lines.append(
        "La vitesse moyenne de remontée sur l'ensemble des "
        f"répétitions est de {mean_ascent:.3f} unités norm./s."
    )

    # --------------------------------------------------------
    # 5. Asyméterie
    # --------------------------------------------------------

    lines.append("")
    lines.append("5. ASYMÉTRIE FRONTALE")
    lines.append("-" * 30)

    mean_amplitude_difference = clinical_df[
        "frontal_amplitude_difference_percent"
    ].mean()

    mean_timing_difference = clinical_df[
        "frontal_peak_timing_difference_cycle_percent"
    ].mean()

    greater_side_counts = clinical_df["greater_deviation_side"].value_counts()

    earlier_side_counts = clinical_df["earlier_peak_side"].value_counts()

    pattern_counts = clinical_df["bilateral_peak_pattern"].value_counts()

    main_greater_side = (
        greater_side_counts.idxmax()
        if not greater_side_counts.empty
        else "not_detected"
    )

    main_earlier_side = (
        earlier_side_counts.idxmax()
        if not earlier_side_counts.empty
        else "not_detected"
    )

    main_pattern = (
        pattern_counts.idxmax() if not pattern_counts.empty else "not_detected"
    )

    lines.append(
        "Différence moyenne d'amplitude entre les côtés : "
        f"{mean_amplitude_difference:.2f} points."
    )

    lines.append(
        "Le côté présentant le plus fréquemment la plus grande "
        "déviation est le côté "
        f"{_format_direction(main_greater_side)}."
    )

    lines.append(
        "Le décalage temporel moyen entre les pics est de "
        f"{mean_timing_difference:.1f} % du cycle."
    )

    lines.append(
        "Le pic apparaît le plus fréquemment en premier du côté "
        f"{_format_direction(main_earlier_side)}."
    )

    lines.append(
        "Le pattern bilatéral le plus fréquent est : "
        f"{_format_direction(main_pattern)}."
    )

    # --------------------------------------------------------
    # 6. Cycle moyen
    # --------------------------------------------------------

    lines.append("")
    lines.append("6. CYCLE MOYEN")
    lines.append("-" * 30)

    if not mean_cycle_summary:
        lines.append("Le cycle moyen n'a pas pu être calculé.")
    else:
        lines.append(
            "Nombre de répétitions incluses : "
            f"{mean_cycle_summary['n_cycles']}."
        )

        lines.append(
            "Le point bas du cycle moyen apparaît à "
            f"{mean_cycle_summary['mean_cycle_bottom_percent']:.1f} % "
            "du cycle."
        )

        lines.append(
            "Durée moyenne du cycle : "
            f"{mean_cycle_summary['mean_cycle_duration_s']:.2f} s "
            f"(ET : {mean_cycle_summary['mean_cycle_duration_sd_s']:.2f} s)."
        )

        lines.append(
            "Durée moyenne de la descente : "
            f"{mean_cycle_summary['mean_descent_duration_s']:.2f} s "
            f"(ET : {mean_cycle_summary['mean_descent_duration_sd_s']:.2f} s)."
        )

        lines.append(
            "Durée moyenne de la remontée : "
            f"{mean_cycle_summary['mean_ascent_duration_s']:.2f} s "
            f"(ET : {mean_cycle_summary['mean_ascent_duration_sd_s']:.2f} s)."
        )

        lines.append(
            "Déplacement vertical maximal moyen du pelvis : "
            f"{mean_cycle_summary['mean_cycle_pelvis_peak_displacement']:.4f} "
            "unité normalisée."
        )

        lines.append(
            "Pic moyen gauche : "
            f"{abs(mean_cycle_summary['mean_cycle_left_peak_signed_deviation_percent']):.2f} % "
            f"en {_format_direction(mean_cycle_summary['mean_cycle_left_peak_direction'])}, "
            f"à {mean_cycle_summary['mean_cycle_left_peak_timing_percent']:.1f} % "
            "du cycle."
        )

        lines.append(
            "Pic moyen droit : "
            f"{abs(mean_cycle_summary['mean_cycle_right_peak_signed_deviation_percent']):.2f} % "
            f"en {_format_direction(mean_cycle_summary['mean_cycle_right_peak_direction'])}, "
            f"à {mean_cycle_summary['mean_cycle_right_peak_timing_percent']:.1f} % "
            "du cycle."
        )

        lines.append(
            "Pic moyen de vitesse de descente : "
            f"{mean_cycle_summary['mean_cycle_peak_descent_velocity']:.4f} "
            "unités norm./s, à "
            f"{mean_cycle_summary['mean_cycle_peak_descent_velocity_timing_percent']:.1f} % "
            "du cycle."
        )

        lines.append(
            "Pic moyen de vitesse de remontée : "
            f"{mean_cycle_summary['mean_cycle_peak_ascent_velocity']:.4f} "
            "unités norm./s, à "
            f"{mean_cycle_summary['mean_cycle_peak_ascent_velocity_timing_percent']:.1f} % "
            "du cycle."
        )

        lines.append(
            "Variabilité moyenne du déplacement du pelvis : "
            f"{mean_cycle_summary['mean_cycle_pelvis_average_variability']:.4f}."
        )

        lines.append(
            "Variabilité moyenne de la déviation gauche : "
            f"{mean_cycle_summary['mean_cycle_left_deviation_average_variability']:.2f} "
            "points de pourcentage."
        )

        lines.append(
            "Variabilité moyenne de la déviation droite : "
            f"{mean_cycle_summary['mean_cycle_right_deviation_average_variability']:.2f} "
            "points de pourcentage."
        )

        lines.append(
            "Variabilité moyenne de la vitesse verticale : "
            f"{mean_cycle_summary['mean_cycle_velocity_average_variability']:.4f} "
            "unités norm./s."
        )

        lines.append("")
        lines.append(
            "Interprétation : ce rapport décrit uniquement les "
            "caractéristiques cinématiques observées en vue frontale. "
            "Il ne constitue pas un diagnostic et devra être complété "
            "par les autres vues et tests fonctionnels."
        )

    return "\n".join(lines)
