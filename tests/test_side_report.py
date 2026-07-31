"""Tests du rapport descriptif sagittal."""

import unittest

from movement_analysis.side_repetition_metrics import (
    build_side_repetition_metrics,
)
from reporting.side_report import build_side_report
from tests.test_side_repetition_metrics import (
    build_processed_dataframe,
    build_repetitions_dataframe,
)


class SideReportTests(unittest.TestCase):
    def test_report_contains_primary_and_secondary_sections(self):
        metrics = build_side_repetition_metrics(
            build_processed_dataframe(),
            build_repetitions_dataframe(),
        )

        report = build_side_report(
            metrics,
            {
                "analysis_side": "left",
                "duration_s": 3.0,
                "visibility_ok_percent": 100.0,
            },
        )

        self.assertIn("ANALYSE SAGITTALE", report)
        self.assertIn("MESURES PRINCIPALES", report)
        self.assertIn("genou : 80.0°", report)
        self.assertIn(
            "Vitesse moyenne du genou — descente : 70.0°/s, "
            "remontée : 70.0°/s",
            report,
        )
        self.assertIn("Vitesse moyenne de descente du genou", report)
        self.assertIn("MESURES SECONDAIRES ET EXPÉRIMENTALES", report)
        self.assertIn("1/1", report)
        self.assertIn("ne constitue pas un diagnostic médical", report)

    def test_empty_report_is_explicit(self):
        metrics = build_side_repetition_metrics(
            build_processed_dataframe(),
            build_repetitions_dataframe().iloc[0:0],
        )

        report = build_side_report(metrics)

        self.assertIn("Aucune répétition valide", report)


if __name__ == "__main__":
    unittest.main()
