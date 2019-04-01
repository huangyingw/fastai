
# coding: utf-8

# ## NLP model creation and training

from fastai.gen_doc.nbdoc import *
from fastai.text import *


# The main thing here is [`RNNLearner`](/text.learner.html#RNNLearner). There are also some utility functions to help create and update text models.

# ## Quickly get a learner

show_doc(language_model_learner)


# The model used is given by `arch` and `config`. It can be:
#
# - an [<code>AWD_LSTM</code>](/text.models.html#AWD_LSTM)([Merity et al.](https://arxiv.org/abs/1708.02182))
# - a [<code>Transformer</code>](/text.models.html#Transformer) decoder ([Vaswani et al.](https://arxiv.org/abs/1706.03762))
# - a [<code>TransformerXL</code>](/text.models.html#TransformerXL) ([Dai et al.](https://arxiv.org/abs/1901.02860))
#
# They each have a default config for language modelling that is in <code>{lower_case_class_name}_lm_config</code> if you want to change the default parameter. At this stage, only the AWD LSTM support `pretrained=True` but we hope to add more pretrained models soon. `drop_mult` is applied to all the dropouts weights of the `config`, `learn_kwargs` are passed to the [`Learner`](/basic_train.html#Learner) initialization.

jekyll_note("Using QRNN (change the flag in the config of the AWD LSTM) requires to have cuda installed (same version as pytorch is using).")


path = untar_data(URLs.IMDB_SAMPLE)
data = TextLMDataBunch.from_csv(path, 'texts.csv')
learn = language_model_learner(data, AWD_LSTM, drop_mult=0.5)


show_doc(text_classifier_learner)


# Here again, the backbone of the model is determined by `arch` and `config`. The input texts are fed into that model by bunch of `bptt` and only the last `max_len` activations are considered. This gives us the backbone of our model. The head then consists of:
# - a layer that concatenates the final outputs of the RNN with the maximum and average of all the intermediate outputs (on the sequence length dimension),
# - blocks of ([`nn.BatchNorm1d`](https://pytorch.org/docs/stable/nn.html#torch.nn.BatchNorm1d), [`nn.Dropout`](https://pytorch.org/docs/stable/nn.html#torch.nn.Dropout), [`nn.Linear`](https://pytorch.org/docs/stable/nn.html#torch.nn.Linear), [`nn.ReLU`](https://pytorch.org/docs/stable/nn.html#torch.nn.ReLU)) layers.
#
# The blocks are defined by the `lin_ftrs` and `drops` arguments. Specifically, the first block will have a number of inputs inferred from the backbone arch and the last one will have a number of outputs equal to data.c (which contains the number of classes of the data) and the intermediate blocks have a number of inputs/outputs determined by `lin_ftrs` (of course a block has a number of inputs equal to the number of outputs of the previous block). The dropouts all have a the same value ps if you pass a float, or the corresponding values if you pass a list. Default is to have an intermediate hidden size of 50 (which makes two blocks model_activation -> 50 -> n_classes) with a dropout of 0.1.

path = untar_data(URLs.IMDB_SAMPLE)
data = TextClasDataBunch.from_csv(path, 'texts.csv')
learn = text_classifier_learner(data, AWD_LSTM, drop_mult=0.5)


show_doc(RNNLearner)


# Handles the whole creation from <code>data</code> and a `model` with a text data using a certain `bptt`. The `split_func` is used to properly split the model in different groups for gradual unfreezing and differential learning rates. Gradient clipping of `clip` is optionally applied. `alpha` and `beta` are all passed to create an instance of [`RNNTrainer`](/callbacks.rnn.html#RNNTrainer). Can be used for a language model or an RNN classifier. It also handles the conversion of weights from a pretrained model as well as saving or loading the encoder.

show_doc(RNNLearner.get_preds)


# If `ordered=True`, returns the predictions in the order of the dataset, otherwise they will be ordered by the sampler (from the longest text to the shortest). The other arguments are passed [`Learner.get_preds`](/basic_train.html#Learner.get_preds).

# ### Loading and saving

show_doc(RNNLearner.load_encoder)


show_doc(RNNLearner.save_encoder)


show_doc(RNNLearner.load_pretrained)


# Opens the weights in the `wgts_fname` of `self.model_dir` and the dictionary in `itos_fname` then adapts the pretrained weights to the vocabulary of the <code>data</code>. The two files should be in the models directory of the `learner.path`.

# ## Utility functions

show_doc(convert_weights)


# Uses the dictionary `stoi_wgts` (mapping of word to id) of the weights to map them to a new dictionary `itos_new` (mapping id to word).

# ## Get predictions

show_doc(LanguageLearner, title_level=3)


show_doc(LanguageLearner.predict)


# If `no_unk=True` the unknown token is never picked. Words are taken randomly with the distribution of probabilities returned by the model. If `min_p` is not `None`, that value is the minimum probability to be considered in the pool of words. Lowering `temperature` will make the texts less randomized.

show_doc(LanguageLearner.beam_search)


# ## Basic functions to get a model

show_doc(get_language_model)


show_doc(get_text_classifier)


# This model uses an encoder taken from the `arch` on `config`. This encoder is fed the sequence by successive bits of size `bptt` and we only keep the last `max_seq` outputs for the pooling layers.
#
# The decoder use a concatenation of the last outputs, a `MaxPooling` of all the outputs and an `AveragePooling` of all the outputs. It then uses a list of `BatchNorm`, `Dropout`, `Linear`, `ReLU` blocks (with no `ReLU` in the last one), using a first layer size of `3*emb_sz` then following the numbers in `n_layers`. The dropouts probabilities are read in `drops`.
#
# Note that the model returns a list of three things, the actual output being the first, the two others being the intermediate hidden states before and after dropout (used by the [`RNNTrainer`](/callbacks.rnn.html#RNNTrainer)). Most loss functions expect one output, so you should use a Callback to remove the other two if you're not using [`RNNTrainer`](/callbacks.rnn.html#RNNTrainer).

# ## Undocumented Methods - Methods moved below this line will intentionally be hidden

# ## New Methods - Please document or move to the undocumented section

show_doc(MultiBatchEncoder.forward)


show_doc(LanguageLearner.show_results)


show_doc(MultiBatchEncoder.concat)


show_doc(MultiBatchEncoder)


show_doc(decode_spec_tokens)


show_doc(MultiBatchEncoder.reset)
