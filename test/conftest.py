import typing
from typing import Any, Dict, Tuple, Union
import dm_env
import pytest
import numpy as np

SKIP_GYM_TESTS = False
SKIP_GYM_MESSAGE = 'gym not installed.'
SKIP_ATARI_TESTS = False
SKIP_ATARI_MESSAGE = ''
SKIP_PETTINGZOO_TESTS = False
SKIP_PETTINGZOO_MESSAGE = ''

try:
    # pylint: disable=g-import-not-at-top
    import gymnasium as gym
    # pylint: enable=g-import-not-at-top
except ModuleNotFoundError:
    SKIP_GYM_TESTS = True

try:
    import ale_py  # pylint: disable=g-import-not-at-top
except ModuleNotFoundError as e:
    SKIP_ATARI_TESTS = True
    SKIP_ATARI_MESSAGE = str(e)
except Exception as e:  # pylint: disable=broad-except
    # This exception is raised by atari_py.get_game_path('pong') if the Atari ROM
    # file has not been installed.
    SKIP_ATARI_TESTS = True
    SKIP_ATARI_MESSAGE = str(e)
    del ale_py
else:
    del ale_py
    
try:
    from pettingzoo.utils.env import AECEnv, ParallelEnv
except ModuleNotFoundError as e:
    SKIP_PETTINGZOO_TESTS = True
    SKIP_PETTINGZOO_MESSAGE = str(e)