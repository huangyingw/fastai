
# coding: utf-8

# ## Implementation of the language models

from fastai.gen_doc.nbdoc import *
from fastai.text import *
from fastai.text.models import *


# [`text.models`](/text.models.html#text.models) module fully implements the encoder for an [AWD-LSTM](https://arxiv.org/pdf/1708.02182.pdf), the [transformer model](https://arxiv.org/abs/1706.03762) and the [transformer XL model](https://arxiv.org/abs/1901.02860). They can then plugged in with a decoder to make a language model, or some classifying layers to make a text classifier.

# ## Language model modules

show_doc(AWD_LSTM, title_level=3)


# The main idea of the article is to use a [RNN](http://www.pnas.org/content/79/8/2554) with dropout everywhere, but in an intelligent way. There is a difference with the usual dropout, which is why you’ll see a [`RNNDropout`](/text.models.awd_lstm.html#RNNDropout) module: we zero things, as is usual in dropout, but we always zero the same thing according to the sequence dimension (which is the first dimension in pytorch). This ensures consistency when updating the hidden state through the whole sentences/articles.
#
# This being given, there are a total four different dropouts in the encoder of the AWD-LSTM:
#
# - the first one, embedding dropout, is applied when we look the ids of our tokens inside the embedding matrix (to transform them from numbers to a vector of float). We zero some lines of it, so random ids are sent to a vector of zeros instead of being sent to their embedding vector. This is the `embed_p` parameter.
# - the second one, input dropout, is applied to the result of the embedding with dropout. We forget random pieces of the embedding matrix (but as stated in the last paragraph, the same ones in the sequence dimension). This is the `input_p` parameter.
# - the third one is the weight dropout. It’s the trickiest to implement as we randomly replace by 0s some weights of the hidden-to-hidden matrix inside the RNN: this needs to be done in a way that ensure the gradients are still computed and the initial weights still updated. This is the `weight_p` parameter.
# - the fourth one is the hidden dropout. It’s applied to the output of one of the layers of the RNN before it’s used as input of the next layer (again same coordinates are zeroed in the sequence dimension). It isn’t applied to the last output (which will get its own dropout in the decoder).This is the `hidden_p` parameter.
#
# The other attributes are `vocab_sz` for the number of tokens in your vocabulary, `emb_sz` for the embedding size, `n_hid` for the hidden size of your inner LSTMs (or QRNNs), `n_layers` the number of layers and `pad_token` for the index of an eventual padding token (1 by default in fastai).
#
# The flag `qrnn=True` replace the inner LSTMs by [QRNNs](https://arxiv.org/abs/1611.01576).

show_doc(AWD_LSTM.reset)


show_doc(Transformer, title_level=3)


# The main idea of this article is to use regular neural net for NLP instead of an RNN, but with lots of attention layers. Intuitively, those attention layers tell the model to pay more interest to this or that world when trying to predict its output.
#
# It starts from embeddings from `vocab_sz` (number of tokens) to `d_model` (which is basically the hidden size throughout the model), and it will look at inputs of size batch_size by `ctx_len` (for context length). We add a positional encoding to the embeddings (since a regular neural net has no idea of the order of words), either learned or coming from [`PositionalEncoding`](/text.models.transformer.html#PositionalEncoding) depending on `learned_pos_enc`. We then have a dropout of `embed_p` followed by `n_layers` blocks of [`MultiHeadAttention`](/text.models.transformer.html#MultiHeadAttention) followed by [`feed_forward`](/text.models.transformer.html#feed_forward).
#
# In the attention we use `n_heads` with each a hidden state of `d_head` (will default to `d_model//n_heads`). If `mask=True`, a mask will make sure no attention is paid to future tokens (which would be cheating when training a language model). If `scale=True`, the attention scores are scaled by a factor `1 / math.sqrt(d_head)`. A dropout of `attn_p` is applied to the attention scores, then the final result get applied a dropout of `resid_p` before being summed to the original input (residual connection before the layer norm).
#
# In feed forward, we have two linear layers from `d_model` to `d_inner` and then back. Those have `bias` if that flag is `True` and a dropout of `ff_p` is applied, after each if `double_drop=True`, or just at the end otherwise. `act` is used in the middle as a non-linearity.

show_doc(TransformerXL, title_level=3)


# TransformerXL is a transformer architecture with a sort of hidden state formed by the results of the intermediate layers on previous tokens. Its size is determined by `mem_len`. By using this context, those models are capable of learning longer dependencies and can also be used for faster text generation at inference: a regular transformer model would have to reexamine the whole of sequence of indexes generated so far, whereas we can feed the new tokens one by one to a transformer XL (like we do with a regular RNN).

show_doc(TransformerXL.reset)


# ## Decoders

show_doc(LinearDecoder, title_level=3)


# Create a the decoder to go on top of an [`RNNCore`](/text.models.awd_lstm.html#RNNCore) encoder and create a language model. `n_hid` is the dimension of the last hidden state of the encoder, `n_out` the size of the output. Dropout of `output_p` is applied. If a `tie_encoder` is passed, it will be used for the weights of the linear layer, that will have `bias` or not.

show_doc(PoolingLinearClassifier, title_level=3)


# The last output, `MaxPooling` of all the outputs and `AvgPooling` of all the outputs are concatenated, then blocks of [`bn_drop_lin`](/layers.html#bn_drop_lin) are stacked, according to the values in [`layers`](/layers.html#layers) and `drops`.

# ## Basic NLP modules

# On top of the pytorch or the fastai [`layers`](/layers.html#layers), the language models use some custom layers specific to NLP.

show_doc(EmbeddingDropout, title_level=3)


# Each row of the embedding matrix has a probability `embed_p` of being replaced by zeros while the others are rescaled accordingly.

enc = nn.Embedding(100, 7, padding_idx=1)
enc_dp = EmbeddingDropout(enc, 0.5)
tst_input = torch.randint(0, 100, (8,))
enc_dp(tst_input)


show_doc(RNNDropout, title_level=3)


dp = RNNDropout(0.3)
tst_input = torch.randn(3, 3, 7)
tst_input, dp(tst_input)


show_doc(WeightDropout, title_level=3)


# Applies dropout of probability `weight_p` to the layers in `layer_names` of `module` in training mode. A copy of those weights is kept so that the dropout mask can change at every batch.

module = nn.LSTM(5, 2)
dp_module = WeightDropout(module, 0.4)
getattr(dp_module.module, 'weight_hh_l0')


# It's at the beginning of a forward pass that the dropout is applied to the weights.

tst_input = torch.randn(4, 20, 5)
h = (torch.zeros(1, 20, 2), torch.zeros(1, 20, 2))
x, h = dp_module(tst_input, h)
getattr(dp_module.module, 'weight_hh_l0')


show_doc(PositionalEncoding, title_level=3)


show_doc(DecoderLayer, title_level=3)


show_doc(MultiHeadAttention, title_level=3)


show_doc(MultiHeadRelativeAttention, title_level=3)


show_doc(SequentialRNN, title_level=3)


show_doc(SequentialRNN.reset)


# Call the `reset` function of [`self.children`](/torch_core.html#children) (if they have one).

show_doc(dropout_mask)


tst_input = torch.randn(3, 3, 7)
dropout_mask(tst_input, (3, 7), 0.3)


# Such a mask is then expanded in the sequence length dimension and multiplied by the input to do an [`RNNDropout`](/text.models.awd_lstm.html#RNNDropout).

show_doc(feed_forward)


# ## Undocumented Methods - Methods moved below this line will intentionally be hidden

show_doc(WeightDropout.forward)


show_doc(EmbeddingDropout.forward)


show_doc(RNNDropout.forward)


show_doc(WeightDropout.reset)


show_doc(PoolingLinearClassifier.forward)


show_doc(LinearDecoder.forward)


# ## New Methods - Please document or move to the undocumented section
