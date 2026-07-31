"""Outils géométriques pour l'analyse sagittale du squat.

Les coordonnées MediaPipe sont exprimées dans le repère image :
- x augmente vers la droite ;
- y augmente vers le bas.

Convention générale
-------------------
``facing_direction="right"`` signifie que le sujet regarde vers la droite de
l'image affichée. ``"left"`` signifie qu'il regarde vers la gauche.

Pour les flexions du genou, de la hanche et du tronc :
- valeur positive : flexion dans le sens du squat ;
- valeur négative : extension au-delà de la position neutre.

Conventions des mesures
-----------------------
- ``compute_joint_internal_angle`` retourne l'angle géométrique interne dans
  l'intervalle [0°, 180°] ;
- ``compute_signed_joint_flexion`` retourne 0° pour une articulation tendue et
  une valeur positive lors de la flexion anatomique, indépendamment du sens
  dans lequel le sujet regarde ;
- ``compute_trunk_flexion`` retourne 0° pour un tronc vertical et une valeur
  positive lorsque le tronc s'incline vers l'avant ;
- ``compute_foot_inclination`` retourne 0° pour un pied horizontal et une
  valeur positive lorsque le talon monte par rapport à l'avant-pied ;
- ``compute_ankle_internal_angle`` mesure l'angle entre l'axe talon–avant-pied
  et le segment cheville–genou. Cette définition est moins sensible à la
  position estimée de la cheville que l'angle avant-pied–cheville–genou ;
- ``compute_ankle_dorsiflexion`` est une variation relative à la posture
  debout : ``angle_neutre - angle_courant``. Une valeur positive correspond à
  une augmentation de la dorsiflexion.

Les angles de flexion du genou et de la hanche sont des mesures géométriques
absolues. Leur éventuelle correction par rapport à une baseline appartient au
pipeline d'analyse temporelle, pas à ce module géométrique.
"""

from __future__ import annotations

from typing import Sequence

import numpy as np


Point2D = Sequence[float] | np.ndarray


def _as_point(point: Point2D) -> np.ndarray:
    """Convertit un point 2D en tableau NumPy de forme (2,)."""

    array = np.asarray(point, dtype=float)

    if array.shape != (2,):
        raise ValueError(
            "Chaque point doit contenir exactement deux coordonnées : (x, y)."
        )

    return array


def _to_cartesian_vector(start: Point2D, end: Point2D) -> np.ndarray:
    """Retourne le vecteur start -> end dans un repère où y pointe vers le haut."""

    start_array = _as_point(start)
    end_array = _as_point(end)

    vector_image = end_array - start_array

    return np.array(
        [
            vector_image[0],
            -vector_image[1],
        ],
        dtype=float,
    )


def _facing_factor(facing_direction: str) -> float:
    """Retourne +1 si le sujet regarde à droite, sinon -1."""

    if facing_direction == "right":
        return 1.0

    if facing_direction == "left":
        return -1.0

    raise ValueError(
        "facing_direction doit être égal à 'left' ou 'right'."
    )


def infer_facing_direction(
    heel: Point2D,
    foot_index: Point2D,
) -> str:
    """Déduit la direction dans l'image à partir de l'axe talon–avant-pied.

    ``right`` signifie que l'avant-pied est à droite du talon dans les
    coordonnées brutes de l'image. Cette convention reste valide même si la
    prévisualisation de la webcam est affichée en miroir.
    """

    heel_array = _as_point(heel)
    foot_index_array = _as_point(foot_index)
    horizontal_difference = foot_index_array[0] - heel_array[0]

    if np.isclose(horizontal_difference, 0.0):
        raise ValueError(
            "Direction indéterminable : le talon et l'avant-pied "
            "ont la même coordonnée horizontale."
        )

    return "right" if horizontal_difference > 0 else "left"


def compute_segment_orientation(
    proximal: Point2D,
    distal: Point2D,
) -> float:
    """Calcule l'orientation d'un segment par rapport à l'horizontale."""

    vector = _to_cartesian_vector(proximal, distal)

    if np.allclose(vector, 0.0):
        return np.nan

    return float(
        np.degrees(
            np.arctan2(
                vector[1],
                vector[0],
            )
        )
    )


def compute_oriented_angle(
    first_vector: Point2D,
    second_vector: Point2D,
) -> float:
    """Calcule l'angle orienté du premier vecteur vers le second."""

    vector_1 = _as_point(first_vector)
    vector_2 = _as_point(second_vector)

    norm_1 = np.linalg.norm(vector_1)
    norm_2 = np.linalg.norm(vector_2)

    if np.isclose(norm_1, 0.0) or np.isclose(norm_2, 0.0):
        return np.nan

    cross_product = (
        vector_1[0] * vector_2[1]
        - vector_1[1] * vector_2[0]
    )
    dot_product = float(np.dot(vector_1, vector_2))

    return float(
        np.degrees(
            np.arctan2(
                cross_product,
                dot_product,
            )
        )
    )


def compute_joint_internal_angle(
    point_a: Point2D,
    vertex: Point2D,
    point_c: Point2D,
) -> float:
    """Calcule l'angle interne A–vertex–C entre 0° et 180°."""

    vector_1 = _to_cartesian_vector(vertex, point_a)
    vector_2 = _to_cartesian_vector(vertex, point_c)

    oriented_angle = compute_oriented_angle(
        vector_1,
        vector_2,
    )

    if not np.isfinite(oriented_angle):
        return np.nan

    return float(abs(oriented_angle))


def compute_signed_joint_flexion(
    point_a: Point2D,
    vertex: Point2D,
    point_c: Point2D,
    facing_direction: str,
) -> float:
    """Calcule une flexion articulaire signée à partir de trois points.

    Ordre des points :
    - genou : hanche, genou, cheville ;
    - hanche : genou, hanche, épaule.
    """

    vector_1 = _to_cartesian_vector(vertex, point_a)
    vector_2 = _to_cartesian_vector(vertex, point_c)

    oriented_angle = compute_oriented_angle(
        vector_1,
        vector_2,
    )

    if not np.isfinite(oriented_angle):
        return np.nan

    flexion_magnitude = 180.0 - abs(oriented_angle)

    if np.isclose(flexion_magnitude, 0.0):
        return 0.0

    anatomical_sign = (
        np.sign(oriented_angle)
        * _facing_factor(facing_direction)
    )

    return float(anatomical_sign * flexion_magnitude)


def compute_trunk_flexion(
    hip: Point2D,
    shoulder: Point2D,
    facing_direction: str,
) -> float:
    """Calcule la flexion signée du tronc par rapport à la verticale."""

    trunk_vector = _to_cartesian_vector(hip, shoulder)

    if np.allclose(trunk_vector, 0.0):
        return np.nan

    forward_component = (
        _facing_factor(facing_direction)
        * trunk_vector[0]
    )
    vertical_component = trunk_vector[1]

    return float(
        np.degrees(
            np.arctan2(
                forward_component,
                vertical_component,
            )
        )
    )


def compute_foot_inclination(
    heel: Point2D,
    foot_index: Point2D,
    facing_direction: str,
) -> float:
    """Calcule l'inclinaison signée du pied par rapport à l'horizontale."""

    foot_vector = _to_cartesian_vector(heel, foot_index)

    if np.allclose(foot_vector, 0.0):
        return np.nan

    forward_component = (
        _facing_factor(facing_direction)
        * foot_vector[0]
    )
    vertical_component = foot_vector[1]

    return float(
        -np.degrees(
            np.arctan2(
                vertical_component,
                forward_component,
            )
        )
    )


def compute_ankle_internal_angle(
    heel: Point2D,
    foot_index: Point2D,
    ankle: Point2D,
    knee: Point2D,
) -> float:
    """Calcule l'angle entre l'axe du pied et le segment cheville–genou."""

    foot_vector = _to_cartesian_vector(heel, foot_index)
    shank_vector = _to_cartesian_vector(ankle, knee)

    oriented_angle = compute_oriented_angle(
        foot_vector,
        shank_vector,
    )

    if not np.isfinite(oriented_angle):
        return np.nan

    return float(abs(oriented_angle))


def compute_relative_angle(
    current_angle: float,
    reference_angle: float,
) -> float:
    """Retourne la variation signée la plus courte autour de la référence."""

    if not np.isfinite(current_angle) or not np.isfinite(reference_angle):
        return np.nan

    difference = (current_angle - reference_angle + 180.0) % 360.0 - 180.0

    return float(difference)


def compute_ankle_dorsiflexion(
    ankle_internal_angle: float,
    neutral_ankle_angle: float,
) -> float:
    """Calcule la dorsiflexion relativement à la baseline debout."""

    if not np.isfinite(ankle_internal_angle):
        return np.nan

    if not np.isfinite(neutral_ankle_angle):
        return np.nan

    return float(neutral_ankle_angle - ankle_internal_angle)
