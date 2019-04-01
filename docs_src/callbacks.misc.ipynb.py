
# coding: utf-8

# ## callbacks.misc

# Miscellaneous callbacks that don't belong to any specific group are to be found here.

from fastai.gen_doc.nbdoc import *
from fastai.callbacks.misc import *


show_doc(StopAfterNBatches)


# [`StopAfterNBatches`](/callbacks.misc.html#StopAfterNBatches)

# There could be various uses for this handy callback.
#
# The initial purpose of it was to be able to quickly check memory requirements for a given set of hyperparamaters like `bs` and `size`.
#
# Since all the required GPU memory is setup during the first batch of the first epoch [see tutorial](https://docs.fast.ai/tutorial.resources.html#gpu-memory-usage-anatomy), it's enough to run just 1-2 batches to measure whether your hyperparameters are right and won't lead to Out-Of-Memory (OOM) errors. So instead of waiting for minutes or hours to just discover that your `bs` or `size` are too large, this callback allows you to do it seconds.
#
# You can deploy it on a specific learner (or fit call) just like with any other callback:
#
# ```
# from fastai.callbacks.misc import StopAfterNBatches
# [...]
# learn = cnn_learner([...])
# learn.callbacks.append(StopAfterNBatches(n_batches=2))
# learn.fit_one_cycle(3, max_lr=1e-2)
# ```
# and it'll either fit into the existing memory or it'll immediately fail with OOM error. You may want to add [ipyexperiments](https://github.com/stas00/ipyexperiments/) to show you the memory usage, including the peak usage.
#
# This is good, but it's cumbersome since you have to change the notebook source code and often you will have multiple learners and fit calls in the same notebook, so here is how to do it globally by placing the following code somewhere on top of your notebook and leaving the rest of your notebook unmodified:
#
# ```
# from fastai.callbacks.misc import StopAfterNBatches
# # True turns the speedup on, False return to normal behavior
# tune = True
# #tune = False
# if tune:
#     defaults.extra_callbacks = [StopAfterNBatches(n_batches=2)]
# else:
#     defaults.extra_callbacks = None
# ```
# When you're done tuning your hyper-parameters, just set `tune` to `False` and re-run the notebook to do true fitting.
#
# Do note that when you run this callback, each fit call will be interrupted resulting in the red colored output - that's just an indication that the normal fit didn't happen, so you shouldn't expect any qualitative results out of it.

# ## Undocumented Methods - Methods moved below this line will intentionally be hidden
