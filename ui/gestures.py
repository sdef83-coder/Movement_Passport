def detect_hand_raise(landmarks, pose_landmark):
    """
    Détecte si une main est levée au-dessus de la tête.
    """

    nose = landmarks[pose_landmark.NOSE.value]
    left_wrist = landmarks[pose_landmark.LEFT_WRIST.value]
    right_wrist = landmarks[pose_landmark.RIGHT_WRIST.value]

    left_hand_up = left_wrist.y < nose.y
    right_hand_up = right_wrist.y < nose.y

    return left_hand_up or right_hand_up
