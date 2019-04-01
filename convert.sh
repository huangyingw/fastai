#!/bin/bash -
SCRIPT=$(realpath "$0")
SCRIPTPATH=$(dirname "$SCRIPT")
cd "$SCRIPTPATH"

find . -type f -name *.ipynb | while read ss
do                                                                      
    jupyter nbconvert --to=python --template=python.tpl "$ss" --output "$(basename $ss).py" 
    sed -i"" '/^$/d' "$ss.py"
    sed -i"" '/^#$/d' "$ss.py"
    autopep8 --in-place "$ss.py"
done
