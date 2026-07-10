import time
import math


class SessionManager:
    """
    Gère le cycle d'une session d'analyse.
    """

    def __init__(self, countdown_duration=5, baseline_duration=3):
        self.state = "WAITING"

        self.countdown_duration = countdown_duration
        self.baseline_duration = baseline_duration

        self.countdown_start_time = None
        self.baseline_start_time = None
        self.recording_start_time = None

    def start_countdown(self):
        self.state = "COUNTDOWN"
        self.countdown_start_time = time.time()

    def update(self):
        if self.state == "COUNTDOWN":
            elapsed = time.time() - self.countdown_start_time

            if elapsed >= self.countdown_duration:
                self.state = "BASELINE"
                self.baseline_start_time = time.time()

        elif self.state == "BASELINE":
            elapsed = time.time() - self.baseline_start_time

            if elapsed >= self.baseline_duration:
                self.state = "RECORDING"
                self.recording_start_time = time.time()

    def stop_recording(self):
        self.state = "FINISHED"

    def is_waiting(self):
        return self.state == "WAITING"

    def is_countdown(self):
        return self.state == "COUNTDOWN"

    def is_baseline(self):
        return self.state == "BASELINE"

    def is_recording(self):
        return self.state == "RECORDING"

    def is_finished(self):
        return self.state == "FINISHED"

    def get_recording_time(self):
        if self.recording_start_time is None:
            return 0
        return time.time() - self.recording_start_time

    def get_countdown_remaining(self):
        if self.countdown_start_time is None:
            return self.countdown_duration

        elapsed = time.time() - self.countdown_start_time
        remaining = self.countdown_duration - elapsed

        return max(0, math.ceil(remaining))

    def get_baseline_remaining(self):
        if self.baseline_start_time is None:
            return self.baseline_duration

        elapsed = time.time() - self.baseline_start_time
        remaining = self.baseline_duration - elapsed

        return max(0, math.ceil(remaining))
