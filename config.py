import sys
from yaml import load
try:
    from yaml import CLoader as Loader
except ImportError:
    from yaml import Loader

cfg = {}

def read_config(f):
    cfg.update(load(f, Loader=Loader))
