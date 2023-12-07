"""
InfoLogger is a class to record the information during experiments.
- such as time, number of generated instances, et al.
"""


class InfoLogger:
    def __init__(self):
        self.total_time = None

    def set_total_time(self, time):
        self.total_time = time



