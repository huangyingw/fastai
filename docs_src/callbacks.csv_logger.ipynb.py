
# coding: utf-8

# ## CSV Logger

from fastai.vision import *
from fastai.gen_doc.nbdoc import *
from fastai.callbacks import *


show_doc(CSVLogger)


# First let's show an example of use, with a training on the usual MNIST dataset.

path = untar_data(URLs.MNIST_TINY)
data = ImageDataBunch.from_folder(path)
learn = Learner(data, simple_cnn((3, 16, 16, 2)), metrics=[accuracy, error_rate], callback_fns=[CSVLogger])


learn.fit(3)


# Training details have been saved in 'history.csv'.

# Note that it only saves float/int metrics, so time currently is not saved. This could be saved but requires changing the recording - you can submit a PR [fixing that](https://forums.fast.ai/t/expand-recorder-to-deal-with-non-int-float-data/41534).

learn.path.ls()


# Note that, as with all [`LearnerCallback`](/basic_train.html#LearnerCallback), you can access the object as an attribute of `learn` after it has been created. Here it's `learn.csv_logger`.

show_doc(CSVLogger.read_logged_file)


learn.csv_logger.read_logged_file()


# Optionally you can set `append=True` to log results of consequent stages of training.

# don't forget to remove the old file
if learn.csv_logger.path.exists(): os.remove(learn.csv_logger.path)


learn = Learner(data, simple_cnn((3, 16, 16, 2)), metrics=[accuracy, error_rate],
                callback_fns=[partial(CSVLogger, append=True)])


# stage-1
learn.fit(3)


# stage-2
learn.fit(3)


learn.csv_logger.read_logged_file()


# ### Calback methods

# You don't call these yourself - they're called by fastai's [`Callback`](/callback.html#Callback) system automatically to enable the class's functionality.

show_doc(CSVLogger.on_train_begin)


show_doc(CSVLogger.on_epoch_end)


show_doc(CSVLogger.on_train_end)


# ## Undocumented Methods - Methods moved below this line will intentionally be hidden

# ## New Methods - Please document or move to the undocumented section
