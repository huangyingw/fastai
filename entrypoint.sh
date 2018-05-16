#!/bin/bash -
SCRIPT=$(realpath "$0")
SCRIPTPATH=$(dirname "$SCRIPT")
cd "$SCRIPTPATH"

/root/anaconda3/envs/fastai/bin/jupyter notebook --allow-root &
source /root/anaconda3/bin/activate fastai
python ./courses/dl1/lesson1.py
while true
do
    :
done
