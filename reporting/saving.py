import os


def save_dataframe_csv(df, results_folder, filename):
    """
    Sauvegarde un DataFrame en CSV dans le dossier results.
    """

    csv_path = os.path.join(results_folder, filename)
    df.to_csv(csv_path, index=False)

    return csv_path


def save_summary_txt(summary, results_folder, filename):
    """
    Sauvegarde un résumé de résultats dans un fichier texte.
    """

    txt_path = os.path.join(results_folder, filename)

    with open(txt_path, "w", encoding="utf-8") as file:
        file.write("Résumé FPPA\n")
        file.write("====================\n\n")

        for key, value in summary.items():
            file.write(f"{key} : {value}\n")

    return txt_path


from datetime import datetime


def create_session_folder(results_folder, test_name):
    timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    session_folder = os.path.join(results_folder, f"{test_name}_{timestamp}")
    os.makedirs(session_folder, exist_ok=True)
    return session_folder


def save_metadata_txt(metadata, results_folder, filename="metadata.txt"):
    txt_path = os.path.join(results_folder, filename)

    with open(txt_path, "w", encoding="utf-8") as file:
        file.write("Métadonnées de session\n")
        file.write("====================\n\n")

        for key, value in metadata.items():
            file.write(f"{key} : {value}\n")

    return txt_path


def save_text_report(
    report_text,
    session_folder,
    filename,
):
    """
    Sauvegarde un rapport texte brut.
    """

    report_path = os.path.join(
        session_folder,
        filename,
    )

    with open(
        report_path,
        "w",
        encoding="utf-8",
    ) as file:
        file.write(report_text)

    return report_path
