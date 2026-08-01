from biomechanics.angles import calculate_angle
from vision.landmarks import get_point


def calculate_fppa(
    landmarks,
    pose_landmark,
    side,
    *,
    image_width,
    image_height,
):
    """
    Calcule le FPPA en 2D à partir des points hanche-genou-cheville.

    landmarks : points détectés par MediaPipe
    pose_landmark : mp_pose.PoseLandmark
    side : "left" ou "right"
    """

    if side == "left":
        hip = get_point(
            landmarks,
            pose_landmark.LEFT_HIP,
            image_width,
            image_height,
        )
        knee = get_point(
            landmarks,
            pose_landmark.LEFT_KNEE,
            image_width,
            image_height,
        )
        ankle = get_point(
            landmarks,
            pose_landmark.LEFT_ANKLE,
            image_width,
            image_height,
        )

    elif side == "right":
        hip = get_point(
            landmarks,
            pose_landmark.RIGHT_HIP,
            image_width,
            image_height,
        )
        knee = get_point(
            landmarks,
            pose_landmark.RIGHT_KNEE,
            image_width,
            image_height,
        )
        ankle = get_point(
            landmarks,
            pose_landmark.RIGHT_ANKLE,
            image_width,
            image_height,
        )

    else:
        raise ValueError("side doit être 'left' ou 'right'")

    fppa = calculate_angle(hip, knee, ankle)

    return fppa
