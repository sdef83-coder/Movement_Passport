import numpy as np


def compute_joint_angle(point_a, vertex, point_c):
    """
    Calcule l'angle A-vertex-C en degrés, entre 0 et 180.
    """

    a = np.asarray(point_a, dtype=float)
    b = np.asarray(vertex, dtype=float)
    c = np.asarray(point_c, dtype=float)

    vector_1 = a - b
    vector_2 = c - b

    norm_1 = np.linalg.norm(vector_1)
    norm_2 = np.linalg.norm(vector_2)

    if norm_1 == 0 or norm_2 == 0:
        return np.nan

    cosine = np.dot(vector_1, vector_2) / (norm_1 * norm_2)
    cosine = np.clip(cosine, -1.0, 1.0)

    return float(np.degrees(np.arccos(cosine)))


def compute_segment_angle_from_vertical(proximal, distal):
    """
    Calcule l'inclinaison absolue d'un segment par rapport
    à la verticale de l'image.

    0° = segment vertical.
    90° = segment horizontal.
    """

    proximal = np.asarray(proximal, dtype=float)
    distal = np.asarray(distal, dtype=float)

    vector = distal - proximal

    if np.linalg.norm(vector) == 0:
        return np.nan

    vertical = np.array([0.0, 1.0])

    cosine = np.dot(vector, vertical) / np.linalg.norm(vector)
    cosine = np.clip(cosine, -1.0, 1.0)

    angle = np.degrees(np.arccos(cosine))

    # On veut une inclinaison entre 0 et 90°.
    if angle > 90:
        angle = 180 - angle

    return float(angle)
