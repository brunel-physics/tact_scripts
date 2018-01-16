# -*- coding: utf-8 -*-

"""
This module parses configuration files.
"""

from __future__ import (absolute_import, division, print_function,
                        unicode_literals)

import sys

from yaml import load

try:
    from yaml import CLoader as Loader
except ImportError:
    from yaml import Loader

cfg = {}


def read_config():
    """
    Read the configuration file supplied at the command line, or listen on
    stdin.

    Parameters
    ----------
    None

    Returns
    -------
    None
    """

    if sys.argv[1] == "--stdin":
        f = sys.stdin
    else:
        f = open(sys.argv[1], 'r')

    cfg.update(load(f, Loader=Loader))
