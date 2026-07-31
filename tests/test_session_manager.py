"""Tests du gestionnaire d'état de session."""

import unittest

from session.session_manager import SessionManager


class SessionManagerTests(unittest.TestCase):
    def test_reset_to_waiting_clears_all_timestamps(self):
        session = SessionManager()
        session.state = "RECORDING"
        session.countdown_start_time = 1.0
        session.baseline_start_time = 2.0
        session.recording_start_time = 3.0

        session.reset_to_waiting()

        self.assertTrue(session.is_waiting())
        self.assertIsNone(session.countdown_start_time)
        self.assertIsNone(session.baseline_start_time)
        self.assertIsNone(session.recording_start_time)


if __name__ == "__main__":
    unittest.main()
