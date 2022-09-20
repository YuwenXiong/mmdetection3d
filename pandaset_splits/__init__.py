# Copyright (c) 2021 Waabi Innovation. All rights reserved.

import pathlib


def splits_dir() -> str:
    """Return splits folder"""
    return str(pathlib.Path(__file__).parent)
