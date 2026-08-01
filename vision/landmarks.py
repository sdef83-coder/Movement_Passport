def get_point(
    landmarks,
    landmark_name,
    image_width=1.0,
    image_height=1.0,
):
    """
    Récupère les coordonnées 2D d'un marqueur MediaPipe.

    landmarks : liste des points détectés
    landmark_name : ex. mp_pose.PoseLandmark.RIGHT_KNEE
    """
    point = landmarks[landmark_name.value]
    return [
        point.x * image_width,
        point.y * image_height,
    ]


def get_visibility(landmarks, landmark_name):
    point = landmarks[landmark_name.value]
    return point.visibility


def get_pelvis_y(landmarks, pose_landmark):
    """
    Calcule la position verticale moyenne du bassin à partir des deux hanches.
    """

    left_hip = landmarks[pose_landmark.LEFT_HIP.value]
    right_hip = landmarks[pose_landmark.RIGHT_HIP.value]

    pelvis_y = (left_hip.y + right_hip.y) / 2

    return pelvis_y
