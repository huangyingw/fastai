
# coding: utf-8

# ## Training tweaks for an RNN

from fastai.gen_doc.nbdoc import *
from fastai.callbacks.rnn import *


from fastai.gen_doc.nbdoc import *
from fastai.callbacks.rnn import *


# This callback regroups a few tweaks to properly train RNNs. They all come from [this article](https://arxiv.org/abs/1708.02182) by Stephen Merity et al.
#
# **Activation Regularization:** on top of weight decay, we apply another form of regularization that is pretty similar and consists in adding to the loss a scaled factor of the sum of all the squares of the outputs (with dropout applied) of the various layers of the RNN. Intuitively, weight decay tries to get the network to learn small weights, this is to get the model to learn to produce smaller activations.
#
# **Temporal Activation Regularization:** lastly, we add to the loss a scaled factor of the sum of the squares of the `h_(t+1) - h_t`, where `h_i` is the output (before dropout is applied) of one layer of the RNN at the time step i (word i of the sentence). This will encourage the model to produce activations that don’t vary too fast between two consecutive words of the sentence.

show_doc(RNNTrainer)


# Create a [`Callback`](/callback.html#Callback) that adds to learner the RNN tweaks for training on data with `bptt`. `alpha` is the scale for AR, `beta` is the scale for TAR.

# ### Callback methods

# You don't call these yourself - they're called by fastai's [`Callback`](/callback.html#Callback) system automatically to enable the class's functionality.

show_doc(RNNTrainer.on_epoch_begin)


show_doc(RNNTrainer.on_loss_begin)


# The fastai RNNs return `last_output` that are tuples of three elements, the true output (that is returned) and the hidden states before and after dropout (which are saved internally for the next function).

show_doc(RNNTrainer.on_backward_begin)


# ## Undocumented Methods - Methods moved below this line will intentionally be hidden

# ## New Methods - Please document or move to the undocumented section
