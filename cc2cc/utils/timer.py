"""
A timer utility for measuring execution time of code blocks.
"""

import time


class Timer:
    """
    A simple timer class to measure execution time of code blocks.
    """

    def __init__(self):
        self.start_time = time.time()
        self.latest_time = self.start_time

    def measure(self):
        """Measure the elapsed time since the timer was started."""
        if self.start_time is None:
            raise RuntimeError("Timer has not been started. Call start() first.")
        elapsed_time = time.time() - self.start_time
        elapsed_time_latest = time.time() - self.latest_time
        self.latest_time = time.time()
        return f"Speed: {elapsed_time:.1f} s/E, Latest: {elapsed_time_latest:.1f} s"
