# -*- coding: utf-8 -*-

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
    try:
        if sys.argv[1] == "--stdin":
            f = sys.stdin
        else:
            f = open(sys.argv[1], 'r')

        cfg.update(load(f, Loader=Loader))
    except IndexError:
        print("Usage: requires input file or --stdin to be specified")
        raise
