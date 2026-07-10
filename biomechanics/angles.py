import numpy as np


def calculate_angle(a, b, c):
    """
    Calcule l'angle au point b à partir de 3 points 2D.

    a : premier point, ex. hanche
    b : point central, ex. genou
    c : troisième point, ex. cheville
    """
    a = np.array(a)
    b = np.array(b)
    c = np.array(c)

    ba = a - b
    bc = c - b

    cos_angle = np.dot(ba, bc) / (np.linalg.norm(ba) * np.linalg.norm(bc))
    cos_angle = np.clip(cos_angle, -1.0, 1.0)

    angle = np.degrees(np.arccos(cos_angle))

    return angle
