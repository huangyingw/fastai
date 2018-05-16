#!/bin/bash -
SCRIPT=$(realpath "$0")
SCRIPTPATH=$(dirname "$SCRIPT")
cd "$SCRIPTPATH"

/root/anaconda3/envs/fastai/bin/jupyter notebook --allow-root
