import numpy as np


def compute_frontal_alignment(
    hip,
    knee,
    ankle,
    side,
    neutral_tolerance_percent=2.0,
):
    """
    Mesure la déviation frontale du genou par rapport à l'axe
    hanche-cheville.

    Convention :
    - indice négatif = valgus ;
    - indice positif = varus ;
    - indice proche de zéro = neutre.

    Parameters
    ----------
    hip, knee, ankle : tuple(float, float)
        Coordonnées 2D normalisées (x, y).

    side : str
        "left" ou "right".

    neutral_tolerance_percent : float
        Zone neutre, exprimée en pourcentage de la longueur
        hanche-cheville. Valeur provisoire à valider scientifiquement.

    Returns
    -------
    alignment : str
        "valgus", "neutral", "varus" ou "not_detected".

    alignment_index : float
        Déviation signée sous forme de ratio de la longueur
        hanche-cheville. Exemple : -0.06 = valgus de 6 %.

    signed_deviation_percent : float
        Même mesure, exprimée en pourcentage signé.

    knee_deviation_percent : float
        Amplitude absolue de la déviation, en pourcentage.
    """

    hip = np.asarray(hip, dtype=float)
    knee = np.asarray(knee, dtype=float)
    ankle = np.asarray(ankle, dtype=float)

    hip_ankle_vector = ankle - hip
    hip_knee_vector = knee - hip

    leg_length = np.linalg.norm(hip_ankle_vector)

    if leg_length == 0 or not np.isfinite(leg_length):
        return "not_detected", np.nan, np.nan, np.nan

    cross_product = (
        hip_ankle_vector[0] * hip_knee_vector[1]
        - hip_ankle_vector[1] * hip_knee_vector[0]
    )

    # Distance perpendiculaire signée du genou à l'axe hanche-cheville
    raw_signed_distance = cross_product / leg_length

    # Convention anatomique déjà validée lors de tes essais
    if side == "left":
        signed_distance = -raw_signed_distance
    elif side == "right":
        signed_distance = raw_signed_distance
    else:
        raise ValueError("side doit être 'left' ou 'right'.")

    # Normalisation par la longueur hanche-cheville
    alignment_index = signed_distance / leg_length

    signed_deviation_percent = alignment_index * 100
    knee_deviation_percent = abs(signed_deviation_percent)

    if signed_deviation_percent < -neutral_tolerance_percent:
        alignment = "valgus"
    elif signed_deviation_percent > neutral_tolerance_percent:
        alignment = "varus"
    else:
        alignment = "neutral"

    return (
        alignment,
        float(alignment_index),
        float(signed_deviation_percent),
        float(knee_deviation_percent),
    )
