"""
This module deals with the configuration file
"""
import os
import yaml

_config = None
_filepath = "config.yaml"


def init():

    global _config, _filepath
    if _config is None:
        with open(_filepath, "r") as stream:
            _config = yaml.load(stream)


def get(var_name):

    global _config
    if _config is None:
        init()
    if var_name in _config:
        return _config[var_name]
    else:
        return None


def set(var_name, var_value):
    global _config
    if _config is None:
        init()
    _config[var_name] = var_value
    save()


def save():
    with open(_filepath, "w") as stream:
        yaml.dump(_config, stream, default_flow_style=False)
