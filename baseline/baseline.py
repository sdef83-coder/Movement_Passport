import numpy as np


class BaselineRecorder:
    """
    Enregistre automatiquement les variables biomécaniques
    pendant la posture de référence.
    """

    def __init__(self):
        self.data = {}

    def add_frame(self, **kwargs):
        """
        Ajoute les variables d'une frame.

        Exemple :
        add_frame(
            pelvis_y=0.42,
            left_fppa=178,
            right_fppa=176,
        )
        """

        for key, value in kwargs.items():

            if key not in self.data:
                self.data[key] = []

            if not np.isnan(value):
                self.data[key].append(value)

    def compute(self):
        """
        Calcule les statistiques de référence.
        """

        baseline = {}

        for key, values in self.data.items():

            if len(values) == 0:
                continue

            baseline[f"{key}_mean"] = float(np.mean(values))
            baseline[f"{key}_std"] = float(np.std(values))
            baseline[f"{key}_n"] = len(values)

        return baseline
