#!/bin/bash -
SCRIPT=$(realpath "$0")
SCRIPTPATH=$(dirname "$SCRIPT")
cd "$SCRIPTPATH"

tools/run-after-git-clone
pip install -e ".[dev]"

conda install -c conda-forge \
    jupytext \
    neovim
