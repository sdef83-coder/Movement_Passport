"""Tests du graphique du cycle moyen sagittal."""

import os
import tempfile
import unittest

from movement_analysis.side_mean_cycle import build_side_mean_cycle
from reporting.side_mean_cycle_plotting import plot_side_mean_cycle
from tests.test_side_mean_cycle import build_mean_cycle_inputs


class SideMeanCyclePlottingTests(unittest.TestCase):
    def test_creates_non_empty_png(self):
        processed, repetitions = build_mean_cycle_inputs()
        mean_cycle = build_side_mean_cycle(processed, repetitions)

        with tempfile.TemporaryDirectory() as temporary_directory:
            figure_path = plot_side_mean_cycle(
                mean_cycle,
                temporary_directory,
            )

            self.assertTrue(os.path.exists(figure_path))
            self.assertGreater(os.path.getsize(figure_path), 10_000)

    def test_none_cycle_creates_no_plot(self):
        with tempfile.TemporaryDirectory() as temporary_directory:
            self.assertIsNone(
                plot_side_mean_cycle(None, temporary_directory)
            )


if __name__ == "__main__":
    unittest.main()
