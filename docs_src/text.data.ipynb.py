
# coding: utf-8

# ## NLP datasets

from fastai.gen_doc.nbdoc import *
from fastai.text import *
from fastai.gen_doc.nbdoc import *


# This module contains the [`TextDataset`](/text.data.html#TextDataset) class, which is the main dataset you should use for your NLP tasks. It automatically does the preprocessing steps described in [`text.transform`](/text.transform.html#text.transform). It also contains all the functions to quickly get a [`TextDataBunch`](/text.data.html#TextDataBunch) ready.

# ## Quickly assemble your data

# You should get your data in one of the following formats to make the most of the fastai library and use one of the factory methods of one of the [`TextDataBunch`](/text.data.html#TextDataBunch) classes:
# - raw text files in folders train, valid, test in an ImageNet style,
# - a csv where some column(s) gives the label(s) and the following one the associated text,
# - a dataframe structured the same way,
# - tokens and labels arrays,
# - ids, vocabulary (correspondence id to word) and labels.
#
# If you are assembling the data for a language model, you should define your labels as always 0 to respect those formats. The first time you create a [`DataBunch`](/basic_data.html#DataBunch) with one of those functions, your data will be preprocessed automatically. You can save it, so that the next time you call it is almost instantaneous.
#
# Below are the classes that help assembling the raw data in a [`DataBunch`](/basic_data.html#DataBunch) suitable for NLP.

show_doc(TextLMDataBunch, title_level=3)


# All the texts in the [`datasets`](/datasets.html#datasets) are concatenated and the labels are ignored. Instead, the target is the next word in the sentence.

show_doc(TextLMDataBunch.create)


show_doc(TextClasDataBunch, title_level=3)


show_doc(TextClasDataBunch.create)


# All the texts are grouped by length (with a bit of randomness for the training set) then padded so that the samples have the same length to get in a batch.

show_doc(TextDataBunch, title_level=3)


jekyll_warn("This class can only work directly if all the texts have the same length.")


# ### Factory methods (TextDataBunch)

# All those classes have the following factory methods.

show_doc(TextDataBunch.from_folder)


# The floders are scanned in `path` with a <code>train</code>, `valid` and maybe `test` folders. Text files in the <code>train</code> and `valid` folders should be places in subdirectories according to their classes (not applicable for a language model). `tokenizer` will be used to parse those texts into tokens.
#
# You can pass a specific `vocab` for the numericalization step (if you are building a classifier from a language model you fine-tuned for instance). kwargs will be split between the [`TextDataset`](/text.data.html#TextDataset) function and to the class initialization, you can precise there parameters such as `max_vocab`, `chunksize`, `min_freq`, `n_labels` (see the [`TextDataset`](/text.data.html#TextDataset) documentation) or `bs`, `bptt` and `pad_idx` (see the sections LM data and classifier data).

show_doc(TextDataBunch.from_csv)


# This method will look for `csv_name`, and optionally a `test` csv file, in `path`. These will be opened with [`header`](https://pandas.pydata.org/pandas-docs/stable/reference/api/pandas.read_csv.html#pandas-read-csv), using `delimiter`. You can specify which are the `text_cols` and `label_cols`; by default a single label column is assumed to come before a single text column. If your csv has no header, you must specify these as indices. If you're training a language model and don't have labels, you must specify the `text_cols`. If there are several `text_cols`, the texts will be concatenated together with an optional field token. If there are several `label_cols`, the labels will be assumed to be one-hot encoded and `classes` will default to `label_cols` (you can ignore that argument for a language model). `label_delim` can be used to specify the separator between multiple labels in a column.
#
# You can pass a `tokenizer` to be used to parse the texts into tokens and/or a specific `vocab` for the numericalization step (if you are building a classifier from a language model you fine-tuned for instance). Otherwise you can specify parameters such as `max_vocab`, `min_freq`, `chunksize` for the Tokenizer and Numericalizer (processors). Other parameters (e.g. `bs`, `val_bs` and `num_workers`, etc.) will be passed to [`LabelLists.databunch()`](/data_block.html#LabelLists.databunch) documentation) (see the LM data and classifier data sections for more info).

show_doc(TextDataBunch.from_df)


# This method will use `train_df`, `valid_df` and optionally `test_df` to build the [`TextDataBunch`](/text.data.html#TextDataBunch) in `path`. You can specify `text_cols` and `label_cols`; by default a single label column comes before a single text column. If you're training a language model and don't have labels, you must specify the `text_cols`. If there are several `text_cols`, the texts will be concatenated together with an optional field token. If there are several `label_cols`, the labels will be assumed to be one-hot encoded and `classes` will default to `label_cols` (you can ignore that argument for a language model).
#
# You can pass a `tokenizer` to be used to parse the texts into tokens and/or a specific `vocab` for the numericalization step (if you are building a classifier from a language model you fine-tuned for instance). Otherwise you can specify parameters such as `max_vocab`, `min_freq`, `chunksize` for the default Tokenizer and Numericalizer (processors). Other parameters (e.g. `bs`, `val_bs` and `num_workers`, etc.) will be passed to [`LabelLists.databunch()`](/data_block.html#LabelLists.databunch) documentation) (see the LM data and classifier data sections for more info).

show_doc(TextDataBunch.from_tokens)


# This function will create a [`DataBunch`](/basic_data.html#DataBunch) from `trn_tok`, `trn_lbls`, `val_tok`, `val_lbls` and maybe `tst_tok`.
#
# You can pass a specific `vocab` for the numericalization step (if you are building a classifier from a language model you fine-tuned for instance). kwargs will be split between the [`TextDataset`](/text.data.html#TextDataset) function and to the class initialization, you can precise there parameters such as `max_vocab`, `chunksize`, `min_freq`, `n_labels`, `tok_suff` and `lbl_suff` (see the [`TextDataset`](/text.data.html#TextDataset) documentation) or `bs`, `bptt` and `pad_idx` (see the sections LM data and classifier data).

show_doc(TextDataBunch.from_ids)


# Texts are already preprocessed into `train_ids`, `train_lbls`, `valid_ids`, `valid_lbls` and maybe `test_ids`. You can specify the corresponding `classes` if applicable. You must specify a `path` and the `vocab` so that the [`RNNLearner`](/text.learner.html#RNNLearner) class can later infer the corresponding sizes in the model it will create. kwargs will be passed to the class initialization.

# ### Load and save

# To avoid losing time preprocessing the text data more than once, you should save and load your [`TextDataBunch`](/text.data.html#TextDataBunch) using [`DataBunch.save`](/basic_data.html#DataBunch.save) and [`load_data`](/basic_data.html#load_data).

show_doc(TextDataBunch.load)


jekyll_warn("This method should only be used to load back `TextDataBunch` saved in v1.0.43 or before, it is now deprecated.")


# ### Example

# Untar the IMDB sample dataset if not already done:

path = untar_data(URLs.IMDB_SAMPLE)
path


# Since it comes in the form of csv files, we will use the corresponding `text_data` method. Here is an overview of what your file you should look like:

pd.read_csv(path / 'texts.csv').head()


# And here is a simple way of creating your [`DataBunch`](/basic_data.html#DataBunch) for language modelling or classification.

data_lm = TextLMDataBunch.from_csv(Path(path), 'texts.csv')
data_clas = TextClasDataBunch.from_csv(Path(path), 'texts.csv')


# ## The TextList input classes

# Behind the scenes, the previous functions will create a training, validation and maybe test [`TextList`](/text.data.html#TextList) that will be tokenized and numericalized (if needed) using [`PreProcessor`](/data_block.html#PreProcessor).

show_doc(Text, title_level=3)


show_doc(TextList, title_level=3)


# `vocab` contains the correspondence between ids and tokens, `pad_idx` is the id used for padding. You can pass a custom `processor` in the `kwargs` to change the defaults for tokenization or numericalization. It should have the following form:

tokenizer = Tokenizer(SpacyTokenizer, 'en')
processor = [TokenizeProcessor(tokenizer=tokenizer, NumericalizeProcessor(max_vocab=30000)]


# See below for all the arguments those tokenizers can take.

show_doc(TextList.label_for_lm)


show_doc(TextList.from_folder)


show_doc(TextList.show_xys)


show_doc(TextList.show_xyzs)


show_doc(OpenFileProcessor, title_level=3)


show_doc(open_text)


show_doc(TokenizeProcessor, title_level=3)


# `tokenizer` is used on bits of `chunksize`. If `mark_fields=True`, add field tokens between each parts of the texts (given when the texts are read in several columns of a dataframe). See more about tokenizers in the [transform documentation](/text.transform.html).

show_doc(NumericalizeProcessor, title_level=3)


# Uses `vocab` for this (if not None), otherwise create one with `max_vocab` and `min_freq` from tokens.

# ## Language Model data

# A language model is trained to guess what the next word is inside a flow of words. We don't feed it the different texts separately but concatenate them all together in a big array. To create the batches, we split this array into `bs` chunks of continuous texts. Note that in all NLP tasks, we don't use the usual convention of sequence length being the first dimension so batch size is the first dimension and sequence length is the second. Here you can read the chunks of texts in lines.

path=untar_data(URLs.IMDB_SAMPLE)
data=TextLMDataBunch.from_csv(path, 'texts.csv')
x, y=next(iter(data.train_dl))
example=x[:15, :15].cpu()
texts=pd.DataFrame([data.train_ds.vocab.textify(l).split(' ') for l in example])
texts


jekyll_warn("If you are used to another convention, beware! fastai always uses batch as a first dimension, even in NLP.")


# This is all done internally when we use [`TextLMDataBunch`](/text.data.html#TextLMDataBunch), by wrapping the dataset in the following pre-loader before calling a [`DataLoader`](https://pytorch.org/docs/stable/data.html#torch.utils.data.DataLoader).

show_doc(LanguageModelPreLoader)


# LanguageModelPreLoader is an internal class uses for training a language model. It takes the sentences passed as a jagged array of numericalised sentences in `dataset` and returns contiguous batches to the pytorch dataloader with batch size `bs` and a sequence length `bptt`.
# - `lengths` can be provided for the jagged training data else lengths is calculated internally
# - `backwards=True` will reverses the sentences.
# - `shuffle=True`, will shuffle the order of the sentences, at the start of each epoch - except the first
#
# The following description is usefull for understanding the implementation of [`LanguageModelPreLoader`](/text.data.html#LanguageModelPreLoader):
# - idx: instance of CircularIndex that indexes items while taking the following into account 1) shuffle, 2) direction of indexing, 3) wraps around to head (reading forward) or tail (reading backwards) of the ragged array as needed in order to fill the last batch(s)
#
# - ro: index of the first rag of each row in the batch to be extract. Returns as index to the next rag to be extracted
#
# - ri: Reading forward: index to the first token to be extracted in the current rag (ro). Reading backwards: one position after the last token to be extracted in the rag
#
# - overlap: overlap between batches is 1, because we only predict the next token
#

# ## Classifier data

# When preparing the data for a classifier, we keep the different texts separate, which poses another challenge for the creation of batches: since they don't all have the same length, we can't easily collate them together in batches. To help with this we use two different techniques:
# - padding: each text is padded with the `PAD` token to get all the ones we picked to the same size
# - sorting the texts (ish): to avoid having together a very long text with a very short one (which would then have a lot of `PAD` tokens), we regroup the texts by order of length. For the training set, we still add some randomness to avoid showing the same batches at every step of the training.
#
# Here is an example of batch with padding (the padding index is 1, and the padding is applied before the sentences start).

path=untar_data(URLs.IMDB_SAMPLE)
data=TextClasDataBunch.from_csv(path, 'texts.csv')
iter_dl=iter(data.train_dl)
_=next(iter_dl)
x, y=next(iter_dl)
x[-10:, :20]


# This is all done internally when we use [`TextClasDataBunch`](/text.data.html#TextClasDataBunch), by using the following classes:

show_doc(SortSampler)


# This pytorch [`Sampler`](https://pytorch.org/docs/stable/data.html#torch.utils.data.Sampler) is used for the validation and (if applicable) the test set.

show_doc(SortishSampler)


# This pytorch [`Sampler`](https://pytorch.org/docs/stable/data.html#torch.utils.data.Sampler) is generally used for the training set.

show_doc(pad_collate)


# This will collate the `samples` in batches while adding padding with `pad_idx`. If `pad_first=True`, padding is applied at the beginning (before the sentence starts) otherwise it's applied at the end.

# ## Undocumented Methods - Methods moved below this line will intentionally be hidden

show_doc(TextList.new)


show_doc(TextList.get)


show_doc(TokenizeProcessor.process_one)


show_doc(TokenizeProcessor.process)


show_doc(OpenFileProcessor.process_one)


show_doc(NumericalizeProcessor.process)


show_doc(NumericalizeProcessor.process_one)


show_doc(TextList.reconstruct)


show_doc(LanguageModelPreLoader.on_epoch_begin)


show_doc(LanguageModelPreLoader.on_epoch_end)


# ## New Methods - Please document or move to the undocumented section

show_doc(LMLabelList)


show_doc(LanguageModelPreLoader.allocate_buffers)


show_doc(LanguageModelPreLoader.CircularIndex.shuffle)


show_doc(LanguageModelPreLoader.fill_row)
