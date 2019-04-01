
# coding: utf-8

# ## Hook callbacks

# This provides both a standalone class and a callback for registering and automatically deregistering [PyTorch hooks](https://pytorch.org/tutorials/beginner/former_torchies/nn_tutorial.html#forward-and-backward-function-hooks), along with some pre-defined hooks. Hooks can be attached to any [`nn.Module`](https://pytorch.org/docs/stable/nn.html#torch.nn.Module), for either the forward or the backward pass.
#
# We'll start by looking at the pre-defined hook [`ActivationStats`](/callbacks.hooks.html#ActivationStats), then we'll see how to create our own.

from fastai.gen_doc.nbdoc import *
from fastai.callbacks.hooks import *
from fastai.train import *
from fastai.vision import *


show_doc(ActivationStats)


# [`ActivationStats`](/callbacks.hooks.html#ActivationStats) saves the layer activations in `self.stats` for all `modules` passed to it. By default it will save activations for *all* modules. For instance:

path = untar_data(URLs.MNIST_SAMPLE)
data = ImageDataBunch.from_folder(path)
#learn = cnn_learner(data, models.resnet18, callback_fns=ActivationStats)
learn = Learner(data, simple_cnn((3, 16, 16, 2)), callback_fns=ActivationStats)
learn.fit(1)


# The saved `stats` is a `FloatTensor` of shape `(2,num_modules,num_batches)`. The first axis is `(mean,stdev)`.

len(learn.data.train_dl), len(learn.activation_stats.modules)


learn.activation_stats.stats.shape


# So this shows the standard deviation (`axis0==1`) of 2th last layer (`axis1==-2`) for each batch (`axis2`):

plt.plot(learn.activation_stats.stats[1][-2].numpy());


# ### Internal implementation

show_doc(ActivationStats.hook)


# ### Callback methods

# You don't call these yourself - they're called by fastai's [`Callback`](/callback.html#Callback) system automatically to enable the class's functionality.

show_doc(ActivationStats.on_train_begin)


show_doc(ActivationStats.on_batch_end)


show_doc(ActivationStats.on_train_end)


show_doc(Hook)


# Registers and manually deregisters a [PyTorch hook](https://pytorch.org/tutorials/beginner/former_torchies/nn_tutorial.html#forward-and-backward-function-hooks). Your `hook_func` will be called automatically when forward/backward (depending on `is_forward`) for your module `m` is run, and the result of that function is placed in `self.stored`.

show_doc(Hook.remove)


# Deregister the hook, if not called already.

show_doc(Hooks)


# Acts as a `Collection` (i.e. `len(hooks)` and `hooks[i]`) and an `Iterator` (i.e. `for hook in hooks`) of a group of hooks, one for each module in `ms`, with the ability to remove all as a group. Use `stored` to get all hook results. `hook_func` and `is_forward` behavior is the same as [`Hook`](/callbacks.hooks.html#Hook). See the source code for [`HookCallback`](/callbacks.hooks.html#HookCallback) for a simple example.

show_doc(Hooks.remove)


# Deregister all hooks created by this class, if not previously called.

# ## Convenience functions for hooks

show_doc(hook_output)


# Function that creates a [`Hook`](/callbacks.hooks.html#Hook) for `module` that simply stores the output of the layer.

show_doc(hook_outputs)


# Function that creates a [`Hook`](/callbacks.hooks.html#Hook) for all passed `modules` that simply stores the output of the layers. For example, the (slightly simplified) source code of [`model_sizes`](/callbacks.hooks.html#model_sizes) is:
#
# ```python
# def model_sizes(m, size):
#     x = m(torch.zeros(1, in_channels(m), *size))
#     return [o.stored.shape for o in hook_outputs(m)]
# ```

show_doc(model_sizes)


show_doc(model_summary)


# This method only works on a [`Learner`](/basic_train.html#Learner) object with `train_ds` in it. If it was created as a result of [`load_learner`](/basic_train.html#load_learner), there is no [`data`](/vision.data.html#vision.data) to run through the model and therefore it's not possible to create such summary.
#
# A sample `summary` looks like:
#
# ```
# ======================================================================
# Layer (type)         Output Shape         Param #    Trainable
# ======================================================================
# Conv2d               [64, 176, 176]       9,408      False
# ______________________________________________________________________
# BatchNorm2d          [64, 176, 176]       128        True
# ______________________________________________________________________
# ReLU                 [64, 176, 176]       0          False
# ______________________________________________________________________
# MaxPool2d            [64, 88, 88]         0          False
# ______________________________________________________________________
# Conv2d               [64, 88, 88]         36,864     False
# ...
# ```
#
# Column definition:
#
# 1. **Layer (type)** is the name of the corresponding [`nn.Module`](https://pytorch.org/docs/stable/nn.html#torch.nn.Module).
#
# 2. **Output Shape** is the shape of the output of the corresponding layer (minus the batch dimension, which is always the same and has no impact on the model params).
#
# 3. **Param #** is the number of weights (and optionally bias), and it will vary for each layer.
#
#    The number of params is calculated differently for each layer type. Here is how it's calculated for some of the most common layer types:
#
#     * Conv: `kernel_size*kernel_size*ch_in*ch_out`
#     * Linear: `(n_in+bias) * n_out`
#     * Batchnorm: `2 * n_out`
#     * Embeddings: `n_embed * emb_sz`
#
# 4. **Trainable** indicates whether a layer is trainable or not.
#
#    * Layers with `0` parameters are always Untrainable (e.g., `ReLU` and `MaxPool2d`).
#    * Other layers are either Trainable or not, usually depending on whether they are frozen or not. See [Discriminative layer training](https://docs.fast.ai/basic_train.html#Discriminative-layer-training).
#
# To better understand this summary it helps to also execute `learn.model` and correlate the two outputs.
#
# Example:
#
# Let's feed to a [`Learner`](/basic_train.html#Learner) a dataset of 3-channel images size 352x352 and look at the model and its summary:
#
# ```
# data.train_ds[0][0].data.shape
# learn = cnn_learner(data, models.resnet34, ...)
# print(learn.model)
# print(learn.summary())
# ```
# Here are the outputs with everything but the relevant to the example lines removed:
#
# ```
# torch.Size([3, 352, 352])
#
#     [...]
#     (0): Conv2d(3, 64, kernel_size=(7, 7), stride=(2, 2), padding=(3, 3), bias=False)
#     (1): BatchNorm2d(64, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
#     [...]
#     (3): MaxPool2d(kernel_size=3, stride=2, padding=1, dilation=1, ceil_mode=False)
#     [...]
#     (conv1): Conv2d(64, 64, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False)
#     (8): Linear(in_features=512, out_features=37, bias=True)
#
#
# ======================================================================
# Layer (type)         Output Shape         Param #    Trainable
# ======================================================================
# Conv2d               [64, 176, 176]       9,408      False
# ______________________________________________________________________
# BatchNorm2d          [64, 176, 176]       128        True
# ______________________________________________________________________
# [...]
# MaxPool2d            [64, 88, 88]         0          False
# ______________________________________________________________________
# Conv2d               [64, 88, 88]         36,864     False
# [...]
# ______________________________________________________________________
# Linear               [37]                 18,981     True
#
# ```
#
# **So let's calculate some params:**
#
# For the `Conv2d` layers, multiply the first 4 numbers from the corresponding layer definition:
#
# ```
# Conv2d(3, 64, kernel_size=(7, 7), ...)
#
# 3*64*7*7 = 9,408
#
# Conv2d(64, 64, kernel_size=(3, 3), ...)
#
# 64*64*3*3 = 36,864
# ```
#
# For the `BatchNorm2d` layer, multiply the first number by 2:
# ```
# BatchNorm2d(64, ...)
# 64*2 = 128
# ```
#
# For `Linear` we multiply the first 2 and include the bias if it's `True`:
#
# ```
# Linear(in_features=512, out_features=37, bias=True)
#
# (512+1)*37 = 18,981
# ```
#
# **Now let's calculate some output shapes:**
#
# We started with 3x352x352 image and run it through this layer:
#
# `Conv2d(3, 64, kernel_size=(7, 7), stride=(2, 2), padding=(3, 3), bias=False)`
#
# How did we get: `[64, 176, 176]`
#
# The number of output channels is `64`, that's the first dimension in the number above. And then our image of `352x352` got convolved into `176x176` because of stride `2x2` (`352/2`).
#
# Then we had:
#
# `MaxPool2d(kernel_size=3, stride=2, padding=1, dilation=1, ceil_mode=False)`
#
# which reduced `[64, 176, 176]` to `[64, 88, 88]` again because of stride 2.
#
# And so on, finishing with:
#
# `Linear(in_features=512, out_features=37, bias=True)`
#
# which reduced everything to just `[37]`.

show_doc(num_features_model)


# It can be useful to get the size of each layer of a model (e.g. for printing a summary, or for generating cross-connections for a [`DynamicUnet`](/vision.models.unet.html#DynamicUnet)), however they depend on the size of the input. This function calculates the layer sizes by passing in a minimal tensor of `size`.

show_doc(dummy_batch)


show_doc(dummy_eval)


show_doc(HookCallback)


# For all `modules`, uses a callback to automatically register a method `self.hook` (that you must define in an inherited class) as a hook. This method must have the signature:
#
# ```python
# def hook(self, m:Model, input:Tensors, output:Tensors)
# ```
#
# If `do_remove` then the hook is automatically deregistered at the end of training. See [`ActivationStats`](/callbacks.hooks.html#ActivationStats) for a simple example of inheriting from this class.

# ### Callback methods

# You don't call these yourself - they're called by fastai's [`Callback`](/callback.html#Callback) system automatically to enable the class's functionality.

show_doc(HookCallback.on_train_begin)


show_doc(HookCallback.on_train_end)


# ## Undocumented Methods - Methods moved below this line will intentionally be hidden

show_doc(HookCallback.remove)


show_doc(Hook.hook_fn)


# ## New Methods - Please document or move to the undocumented section
