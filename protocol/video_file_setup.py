"""Sélection et validation d'une vidéo enregistrée."""

from __future__ import annotations

from collections.abc import Callable
from pathlib import Path


SUPPORTED_VIDEO_EXTENSIONS = {
    ".avi",
    ".m4v",
    ".mkv",
    ".mov",
    ".mp4",
}


def validate_video_path(path_value: str | Path) -> Path:
    """Retourne un chemin vidéo absolu ou lève une erreur explicite."""

    normalized_value = str(path_value).strip().strip('"').strip("'")

    if not normalized_value:
        raise ValueError("Aucun fichier vidéo n'a été sélectionné.")

    video_path = Path(normalized_value).expanduser().resolve()

    if not video_path.exists():
        raise ValueError(f"Fichier introuvable : {video_path}")

    if not video_path.is_file():
        raise ValueError(f"Le chemin n'est pas un fichier : {video_path}")

    if video_path.suffix.lower() not in SUPPORTED_VIDEO_EXTENSIONS:
        supported = ", ".join(sorted(SUPPORTED_VIDEO_EXTENSIONS))
        raise ValueError(
            "Format vidéo non pris en charge. "
            f"Formats acceptés : {supported}."
        )

    return video_path


def _open_video_dialog() -> str:
    """Ouvre le sélecteur Windows, avec un retour vide en cas d'indisponibilité."""

    try:
        import tkinter as tk
        from tkinter import filedialog

        root = tk.Tk()
        root.withdraw()
        root.attributes("-topmost", True)

        try:
            return filedialog.askopenfilename(
                title="Choisir la vidéo du squat sagittal",
                filetypes=(
                    ("Vidéos", "*.mp4 *.mov *.m4v *.avi *.mkv"),
                    ("Tous les fichiers", "*.*"),
                ),
            )
        finally:
            root.destroy()
    except Exception:
        # La boîte de dialogue n'est qu'un confort. Dans un environnement sans
        # interface graphique, le parcours continue dans la console Spyder.
        return ""


def choose_video_file(
    input_function: Callable[[str], str] = input,
    dialog_function: Callable[[], str] | None = None,
) -> Path:
    """Demande une vidéo avec une boîte de dialogue puis un repli console."""

    if dialog_function is None:
        dialog_function = _open_video_dialog

    selected_path = dialog_function()

    if selected_path:
        try:
            return validate_video_path(selected_path)
        except ValueError as error:
            print(error)

    print("\nSélection manuelle de la vidéo")
    print("Tu peux copier-coller le chemin complet du fichier.")

    while True:
        path_value = input_function("Chemin de la vidéo : ")

        try:
            return validate_video_path(path_value)
        except ValueError as error:
            print(error)


def choose_baseline_start(
    input_function: Callable[[str], str] = input,
    default_start_s: float = 0.0,
) -> float:
    """Demande à quel instant commence la posture debout de référence."""

    if default_start_s < 0:
        raise ValueError("default_start_s doit être positif ou nul.")

    prompt = (
        "Début de la posture debout dans la vidéo, en secondes "
        f"[{default_start_s:g}] : "
    )

    while True:
        answer = str(input_function(prompt)).strip().replace(",", ".")

        if not answer:
            return float(default_start_s)

        try:
            start_s = float(answer)
        except ValueError:
            print("Entre un nombre de secondes, par exemple 0 ou 2.5.")
            continue

        if start_s < 0:
            print("Le début de la baseline ne peut pas être négatif.")
            continue

        return start_s
