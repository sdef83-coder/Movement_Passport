import cv2
import pandas as pd


def display_fppa_results(image, left_fppa, right_fppa):
    """
    Affiche les résultats FPPA en haut à gauche de l'image.
    """
    if pd.isna(left_fppa) or pd.isna(right_fppa):
        return

    cv2.rectangle(image, (10, 10), (330, 90), (0, 0, 0), -1)

    cv2.putText(
        image,
        f"FPPA gauche : {round(left_fppa, 1)} deg",
        (20, 40),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.7,
        (255, 255, 255),
        2,
    )

    cv2.putText(
        image,
        f"FPPA droit : {round(right_fppa, 1)} deg",
        (20, 75),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.7,
        (255, 255, 255),
        2,
    )


def display_quality_warning(image, visibility_check):
    if visibility_check != "OK":
        cv2.putText(
            image,
            "Donnees non fiables",
            (20, 125),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.7,
            (255, 255, 255),
            2,
        )


def display_instructions(image, camera_view="front"):
    view_label = "Vue laterale" if camera_view == "side" else "Vue de face"
    cv2.putText(
        image,
        f"{view_label} - Corps entier visible - Q pour quitter",
        (20, image.shape[0] - 20),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.6,
        (255, 255, 255),
        2,
    )


def display_session_state(image, session):
    if session.is_waiting():
        text = "Place-toi puis appuie sur S"

    elif session.is_countdown():
        text = f"Debut dans {session.get_countdown_remaining()}"

    elif session.is_baseline():
        text = f"Reste immobile - baseline {session.get_baseline_remaining()}"

    elif session.is_recording():
        text = "ENREGISTREMENT EN COURS - Q pour arreter"

    elif session.is_finished():
        text = "Session terminee"

    else:
        text = "Etat inconnu"

    cv2.putText(
        image,
        text,
        (20, 40),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.8,
        (255, 255, 255),
        2,
    )


def display_frontal_alignment(
    image,
    left_alignment,
    right_alignment,
    left_alignment_index,
    right_alignment_index,
):
    cv2.putText(
        image,
        f"Gauche : {left_alignment} ({left_alignment_index:.3f})",
        (20, 110),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.65,
        (255, 255, 255),
        2,
    )

    cv2.putText(
        image,
        f"Droite : {right_alignment} ({right_alignment_index:.3f})",
        (20, 145),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.65,
        (255, 255, 255),
        2,
    )


def display_frontal_alignment_status(
    image,
    left_alignment,
    right_alignment,
):
    """
    Affiche uniquement l'état d'alignement frontal
    et colore le contour de l'image.

    Priorité :
    valgus > varus > neutral
    """

    alignments = {left_alignment, right_alignment}

    if "not_detected" in alignments:
        border_color = (255, 255, 255)
        global_status = "NON DETECTE"

    if "valgus" in alignments:
        border_color = (0, 0, 255)  # rouge en BGR
        global_status = "VALGUS"

    elif "varus" in alignments:
        border_color = (0, 255, 0)  # vert
        global_status = "VARUS"

    else:
        border_color = (220, 220, 220)  # gris clair
        global_status = "NEUTRE"

    height, width = image.shape[:2]

    # Contour global
    cv2.rectangle(
        image,
        (3, 3),
        (width - 4, height - 4),
        border_color,
        8,
    )

    # Fond noir pour rendre le texte lisible
    cv2.rectangle(
        image,
        (15, 15),
        (360, 115),
        (0, 0, 0),
        -1,
    )

    cv2.putText(
        image,
        f"Gauche : {left_alignment.upper()}",
        (30, 50),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.7,
        border_color,
        2,
    )

    cv2.putText(
        image,
        f"Droite : {right_alignment.upper()}",
        (30, 82),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.7,
        border_color,
        2,
    )

    cv2.putText(
        image,
        f"Etat global : {global_status}",
        (30, 108),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.65,
        border_color,
        2,
    )
