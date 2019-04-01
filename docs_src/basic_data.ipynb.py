
# coding: utf-8

# ## Get your data ready for training

# This module defines the basic [`DataBunch`](/basic_data.html#DataBunch) object that is used inside [`Learner`](/basic_train.html#Learner) to train a model. This is the generic class, that can take any kind of fastai [`Dataset`](https://pytorch.org/docs/stable/data.html#torch.utils.data.Dataset) or [`DataLoader`](https://pytorch.org/docs/stable/data.html#torch.utils.data.DataLoader). You'll find helpful functions in the data module of every application to directly create this [`DataBunch`](/basic_data.html#DataBunch) for you.

from fastai.gen_doc.nbdoc import *
from fastai.basics import *


show_doc(DataBunch)


# It also ensures all the dataloaders are on `device` and applies to them `dl_tfms` as batch are drawn (like normalization). `path` is used internally to store temporary files, `collate_fn` is passed to the pytorch `Dataloader` (replacing the one there) to explain how to collate the samples picked for a batch. By default, it applies data to the object sent (see in [`vision.image`](/vision.image.html#vision.image) or the [data block API](/data_block.html) why this can be important).
#
# `train_dl`, `valid_dl` and optionally `test_dl` will be wrapped in [`DeviceDataLoader`](/basic_data.html#DeviceDataLoader).

# ### Factory method

show_doc(DataBunch.create)


# `num_workers` is the number of CPUs to use, `tfms`, `device` and `collate_fn` are passed to the init method.

jekyll_warn("You can pass regular pytorch Dataset here, but they'll require more attributes than the basic ones to work with the library. See below for more details.")


# ### Visualization

show_doc(DataBunch.show_batch)


# ### Grabbing some data

show_doc(DataBunch.dl)


show_doc(DataBunch.one_batch)


show_doc(DataBunch.one_item)


show_doc(DataBunch.sanity_check)


# ### Load and save

# You can save your [`DataBunch`](/basic_data.html#DataBunch) object for future use with this method.

show_doc(DataBunch.save)


show_doc(load_data)


jekyll_important("The arguments you passed when you created your first `DataBunch` aren't saved, so you should pass them here if you don't want the default.")


# This is to allow you to easily create a new [`DataBunch`](/basic_data.html#DataBunch) with a different batch size for instance. You will also need to reapply any normalization (in vision) you might have done on your original [`DataBunch`](/basic_data.html#DataBunch).

# ### Empty [`DataBunch`](/basic_data.html#DataBunch) for inference

show_doc(DataBunch.export)


show_doc(DataBunch.load_empty, full_name='load_empty')


# This method should be used to create a [`DataBunch`](/basic_data.html#DataBunch) at inference, see the corresponding [tutorial](/tutorial.inference.html).

show_doc(DataBunch.add_test)


# ### Dataloader transforms

show_doc(DataBunch.add_tfm)


# Adds a transform to all dataloaders.

# ## Using a custom Dataset in fastai

# If you want to use your pytorch [`Dataset`](https://pytorch.org/docs/stable/data.html#torch.utils.data.Dataset) in fastai, you may need to implement more attributes/methods if you want to use the full functionality of the library. Some functions can easily be used with your pytorch [`Dataset`](https://pytorch.org/docs/stable/data.html#torch.utils.data.Dataset) if you just add an attribute, for others, the best would be to create your own [`ItemList`](/data_block.html#ItemList) by following [this tutorial](/tutorial.itemlist.html). Here is a full list of what the library will expect.

# ### Basics

# First of all, you obviously need to implement the methods `__len__` and `__getitem__`, as indicated by the pytorch docs. Then the most needed things would be:
# - `c` attribute: it's used in most functions that directly create a [`Learner`](/basic_train.html#Learner) ([`tabular_learner`](/tabular.data.html#tabular_learner), [`text_classifier_learner`](/text.learner.html#text_classifier_learner), [`unet_learner`](/vision.learner.html#unet_learner), [`cnn_learner`](/vision.learner.html#cnn_learner)) and represents the number of outputs of the final layer of your model (also the number of classes if applicable).
# - `classes` attribute: it's used by [`ClassificationInterpretation`](/train.html#ClassificationInterpretation) and also in [`collab_learner`](/collab.html#collab_learner) (best to use [`CollabDataBunch.from_df`](/collab.html#CollabDataBunch.from_df) than a pytorch [`Dataset`](https://pytorch.org/docs/stable/data.html#torch.utils.data.Dataset)) and represents the unique tags that appear in your data.
# - maybe a `loss_func` attribute: that is going to be used by [`Learner`](/basic_train.html#Learner) as a default loss function, so if you know your custom [`Dataset`](https://pytorch.org/docs/stable/data.html#torch.utils.data.Dataset) requires a particular loss, you can put it.
#

# ### For a specific application

# In text, your dataset will need to have a `vocab` attribute that should be an instance of [`Vocab`](/text.transform.html#Vocab). It's used by [`text_classifier_learner`](/text.learner.html#text_classifier_learner) and [`language_model_learner`](/text.learner.html#language_model_learner) when building the model.
#
# In tabular, your dataset will need to have a `cont_names` attribute (for the names of continuous variables) and a `get_emb_szs` method that returns a list of tuple `(n_classes, emb_sz)` representing, for each categorical variable, the number of different codes (don't forget to add 1 for nan) and the corresponding embedding size. Those two are used with the `c` attribute by [`tabular_learner`](/tabular.data.html#tabular_learner).

# ### Functions that really won't work

# To make those last functions work, you really need to use the [data block API](/data_block.html) and maybe write your own [custom ItemList](/tutorial.itemlist.html).

# - [`DataBunch.show_batch`](/basic_data.html#DataBunch.show_batch) (requires `.x.reconstruct`, `.y.reconstruct` and `.x.show_xys`)
# - [`Learner.predict`](/basic_train.html#Learner.predict) (requires `x.set_item`, `.y.analyze_pred`, `.y.reconstruct` and maybe `.x.reconstruct`)
# - [`Learner.show_results`](/basic_train.html#Learner.show_results) (requires `x.reconstruct`, `y.analyze_pred`, `y.reconstruct` and `x.show_xyzs`)
# - `DataBunch.set_item` (requires `x.set_item`)
# - [`Learner.backward`](/basic_train.html#Learner.backward) (uses `DataBunch.set_item`)
# - [`DataBunch.export`](/basic_data.html#DataBunch.export) (requires `export`)

show_doc(DeviceDataLoader)


# Put the batches of `dl` on `device` after applying an optional list of `tfms`. `collate_fn` will replace the one of `dl`. All dataloaders of a [`DataBunch`](/basic_data.html#DataBunch) are of this type.

# ### Factory method

show_doc(DeviceDataLoader.create)


# The given `collate_fn` will be used to put the samples together in one batch (by default it grabs their data attribute). `shuffle` means the dataloader will take the samples randomly if that flag is set to `True`, or in the right order otherwise. `tfms` are passed to the init method. All `kwargs` are passed to the pytorch [`DataLoader`](https://pytorch.org/docs/stable/data.html#torch.utils.data.DataLoader) class initialization.

# ### Methods

show_doc(DeviceDataLoader.add_tfm)


show_doc(DeviceDataLoader.remove_tfm)


show_doc(DeviceDataLoader.new)


show_doc(DeviceDataLoader.proc_batch)


show_doc(DatasetType, doc_string=False)


# Internal enumerator to name the training, validation and test dataset/dataloader.

# ## Undocumented Methods - Methods moved below this line will intentionally be hidden

show_doc(DeviceDataLoader.collate_fn)


# ## New Methods - Please document or move to the undocumented section
