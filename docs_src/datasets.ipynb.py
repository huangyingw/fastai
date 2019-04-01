
# coding: utf-8

# ## datasets

# This module has the necessary functions to be able to download several useful datasets that we might be interested in using in our models.

from fastai.gen_doc.nbdoc import *
from fastai.datasets import *
from fastai.datasets import Config
from pathlib import Path


show_doc(URLs)


# This contains all the datasets' and models' URLs, and some classmethods to help use them - you don't create objects of this class. The supported datasets are (with their calling name): `S3_NLP`, `S3_COCO`, `MNIST_SAMPLE`, `MNIST_TINY`, `IMDB_SAMPLE`, `ADULT_SAMPLE`, `ML_SAMPLE`, `PLANET_SAMPLE`, `CIFAR`, `PETS`, `MNIST`. To get details on the datasets you can see the [fast.ai datasets webpage](http://course.fast.ai/datasets). Datasets with SAMPLE in their name are subsets of the original datasets. In the case of MNIST, we also have a TINY dataset which is even smaller than MNIST_SAMPLE.
#
# Models is now limited to `WT103` but you can expect more in the future!

URLs.MNIST_SAMPLE


# ## Downloading Data

# For the rest of the datasets you will need to download them with [`untar_data`](/datasets.html#untar_data) or [`download_data`](/datasets.html#download_data). [`untar_data`](/datasets.html#untar_data) will decompress the data file and download it while [`download_data`](/datasets.html#download_data) will just download and save the compressed file in `.tgz` format.
#
# The locations where the data and models are downloaded are set in `config.yml`, which by default is located in `~/.fastai`. This directory can be changed via the optional environment variable `FASTAI_HOME` (e.g FASTAI_HOME=/home/.fastai).
#
# If no `config.yml` is present in the specified directory, a default one will be created with `data_path` and `models_path` entries pointing to respectively [`data`](/tabular.data.html#tabular.data) folder and [`models`](/tabular.models.html#tabular.models) folder in the same directory as `config.yml`.
#
# Configure those download locations by editing `data_path` and `models_path` in `config.yml`.

show_doc(untar_data)


untar_data(URLs.PLANET_SAMPLE)


show_doc(download_data)


# Note: If the data file already exists in a <code>data</code> directory inside the notebook, that data file will be used instead of the one present in the folder specified in `config.yml`. `config.yml` is located in the directory specified in optional environment variable `FASTAI_HOME` (defaults to `~/.fastai/`). Paths are resolved by calling the function [`datapath4file`](/datasets.html#datapath4file) - which checks if data exists locally (`data/`) first, before downloading to the folder specified in `config.yml`.
#
# Example:

download_data(URLs.PLANET_SAMPLE)


show_doc(datapath4file)


# All the downloading functions use this to decide where to put the tgz and expanded folder. If `filename` already exists in a <code>data</code> directory in the same place as the calling notebook/script, that is used as the parent directly, otherwise `config.yml` is read to see what path to use, which defaults to <code>~/.fastai/data</code> is used. To override this default, simply modify the value in your `config.yml`:
#
#     data_path: ~/.fastai/data
#
# `config.yml` is located in the directory specified in optional environment variable `FASTAI_HOME` (defaults to `~/.fastai/`).

show_doc(url2path)


show_doc(Config)


# You probably won't need to use this yourself - it's used by `URLs.datapath4file`.

show_doc(Config.get_path)


# Get the key corresponding to `path` in the [`Config`](/datasets.html#Config).

show_doc(Config.data_path)


# Get the `Path` where the data is stored.

show_doc(Config.model_path)


# ## Undocumented Methods - Methods moved below this line will intentionally be hidden

show_doc(Config.create)


show_doc(url2name)


show_doc(Config.get_key)


show_doc(Config.get)


# ## New Methods - Please document or move to the undocumented section
