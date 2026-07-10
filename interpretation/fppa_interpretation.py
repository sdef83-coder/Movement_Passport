def interpret_fppa(left_fppa, right_fppa):
    """
    Interprétation simple du FPPA.

    Attention :
    Ces seuils sont provisoires.
    Ils servent uniquement à structurer le code.
    Ils devront être remplacés par des seuils issus de la littérature.
    """

    interpretation = {
        "left_status": "Non évalué",
        "right_status": "Non évalué",
        "global_message": "Interprétation non disponible",
    }

    if left_fppa >= 170:
        interpretation["left_status"] = "Bon alignement"
    else:
        interpretation["left_status"] = "Alignement à surveiller"

    if right_fppa >= 170:
        interpretation["right_status"] = "Bon alignement"
    else:
        interpretation["right_status"] = "Alignement à surveiller"

    if (
        interpretation["left_status"] == "Bon alignement"
        and interpretation["right_status"] == "Bon alignement"
    ):
        interpretation["global_message"] = (
            "Contrôle frontal du genou satisfaisant"
        )
    else:
        interpretation["global_message"] = (
            "Contrôle frontal du genou à surveiller"
        )

    return interpretation
