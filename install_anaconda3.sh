#!/bin/bash -
SCRIPT=$(realpath "$0")
SCRIPTPATH=$(dirname "$SCRIPT")
cd "$SCRIPTPATH"

# Install Anaconda3
curl -so ~/anaconda3.sh https://repo.anaconda.com/archive/Anaconda3-5.1.0-Linux-x86_64.sh \
    && chmod +x ~/anaconda3.sh \
    && ~/anaconda3.sh -b -p ~/anaconda3 \
    && rm ~/anaconda3.sh
