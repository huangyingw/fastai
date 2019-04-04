
# coding: utf-8

# ## Modify Display Utils

# Utilities for collecting/checking [`fastai`](/fastai.html#fastai) user environment

from fastai.utils.mod_display import *


from fastai.gen_doc.nbdoc import *
from fastai.utils.collect_env import *


show_doc(progress_disabled_ctx)


from fastai.vision import *
path = untar_data(URLs.MNIST_SAMPLE)
data = ImageDataBunch.from_folder(path)
learn = cnn_learner(data, models.resnet18, metrics=accuracy)


# `learn.fit()` will display a progress bar and give the final results once completed:

learn.fit(1)


# [`progress_disabled_ctx`](/utils.mod_display.html#progress_disabled_ctx) will remove all that update and only show the total time once completed.

with progress_disabled_ctx(learn) as learn:
    learn.fit(1)


# ## Undocumented Methods - Methods moved below this line will intentionally be hidden

# ## New Methods - Please document or move to the undocumented section

show_doc(progress_disabled_ctx)
