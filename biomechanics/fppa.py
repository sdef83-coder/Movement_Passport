from biomechanics.angles import calculate_angle
from vision.landmarks import get_point


def calculate_fppa(landmarks, pose_landmark, side):
    """
    Calcule le FPPA en 2D à partir des points hanche-genou-cheville.

    landmarks : points détectés par MediaPipe
    pose_landmark : mp_pose.PoseLandmark
    side : "left" ou "right"
    """

    if side == "left":
        hip = get_point(landmarks, pose_landmark.LEFT_HIP)
        knee = get_point(landmarks, pose_landmark.LEFT_KNEE)
        ankle = get_point(landmarks, pose_landmark.LEFT_ANKLE)

    elif side == "right":
        hip = get_point(landmarks, pose_landmark.RIGHT_HIP)
        knee = get_point(landmarks, pose_landmark.RIGHT_KNEE)
        ankle = get_point(landmarks, pose_landmark.RIGHT_ANKLE)

    else:
        raise ValueError("side doit être 'left' ou 'right'")

    fppa = calculate_angle(hip, knee, ankle)

    return fppa
