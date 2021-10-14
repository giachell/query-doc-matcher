#
# IcLogger
#
# Author: Fabio Giachelle <giachelle.fabio@gmail.com>
# URL: <http://nltk.org/>
# License: MIT

"""
IcLogger

This module provides logging functions leveraging on the https://github.com/gruns/icecream python library.

"""

from icecream import ic


class IcLogger:
    def __init__(self, print_status=True):
        self.print_status = print_status

        if not print_status:
            ic.disable()
        else:
            ic.enable()

    def log(self, *args):
        if self.print_status:
            ic.enable()
            ic(args)
            ic.disable()

    def get_status(self):
        return self.print_status

    def set_status(self, status):
        self.print_status = status

    @staticmethod
    def print_always(*args):
        ic.enable()
        ic(args)
        ic.disable()
