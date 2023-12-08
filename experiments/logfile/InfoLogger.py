"""
InfoLogger is a class to record the information during experiments.
- such as time, number of generated instances, et al.
"""


class InfoLogger:
    def __init__(self):
        # time
        self.total_time = None
        self.global_time = None
        self.local_time = None
        # instance number
        self.all_number = None
        self.global_number = None
        self.local_number = None
        self.all_non_duplicate_number = None
        self.global_non_duplicate_number = None
        self.local_non_duplicate_number = None

    def set_total_time(self, time):
        self.total_time = time

    def set_global_time(self, time):
        self.global_time = time

    def set_local_time(self, time):
        self.local_time = time

    def set_all_number(self, number):
        self.all_number = number

    def set_global_number(self, number):
        self.global_number = number

    def set_local_number(self, number):
        self.local_number = number

    def set_all_non_duplicate_number(self, number):
        self.all_non_duplicate_number = number

    def set_global_non_duplicate_number(self, number):
        self.global_non_duplicate_number = number

    def set_local_non_duplicate_number(self, number):
        self.local_non_duplicate_number = number
