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
        self.step = 0

    def measure(self):
        """Measure the elapsed time since the timer was started."""
        if self.start_time is None:
            raise RuntimeError("Timer has not been started. Call start() first.")
        elapsed_time_latest = time.time() - self.latest_time
        self.latest_time = time.time()
        self.step += 1
        return f"Latest: {elapsed_time_latest:>6.1f} s"

    def measure_all(self):
        """Measure the elapsed time since the timer was started."""
        if self.start_time is None:
            raise RuntimeError("Timer has not been started. Call start() first.")
        elapsed_time = time.time() - self.start_time
        elapsed_time_latest = time.time() - self.latest_time
        self.latest_time = time.time()
        self.step += 1
        return f"Speed: {(elapsed_time / self.step):>6.1f} s/E, Latest: {elapsed_time_latest:>6.1f} s"

    def reset(self):
        """Reset the timer."""
        self.start_time = time.time()
        self.latest_time = self.start_time
        self.step = 0

    def elapsed(self):
        """Get the elapsed time."""
        if self.start_time is None:
            raise RuntimeError("Timer has not been started. Call start() first.")
        return time.time() - self.start_time
