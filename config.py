from yaml import load
try:
    from yaml import CLoader as Loader
except ImportError:
    from yaml import Loader

cfg = {}

def read_config(config_file):
    with open(config_file, 'r') as f:
        cfg.update(load(f, Loader=Loader))
