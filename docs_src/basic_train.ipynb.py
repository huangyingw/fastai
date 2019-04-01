
# coding: utf-8

# ## Basic training functionality

from fastai.basic_train import *
from fastai.gen_doc.nbdoc import *
from fastai.vision import *
from fastai.distributed import *


# [`basic_train`](/basic_train.html#basic_train) wraps together the data (in a [`DataBunch`](/basic_data.html#DataBunch) object) with a PyTorch model to define a [`Learner`](/basic_train.html#Learner) object. Here the basic training loop is defined for the [`fit`](/basic_train.html#fit) method. The [`Learner`](/basic_train.html#Learner) object is the entry point of most of the [`Callback`](/callback.html#Callback) objects that will customize this training loop in different ways. Some of the most commonly used customizations are available through the [`train`](/train.html#train) module, notably:
#
#  - [`Learner.lr_find`](/train.html#lr_find) will launch an LR range test that will help you select a good learning rate.
#  - [`Learner.fit_one_cycle`](/train.html#fit_one_cycle) will launch a training using the 1cycle policy to help you train your model faster.
#  - [`Learner.to_fp16`](/train.html#to_fp16) will convert your model to half precision and help you launch a training in mixed precision.

show_doc(Learner, title_level=2)


# The main purpose of [`Learner`](/basic_train.html#Learner) is to train `model` using [`Learner.fit`](/basic_train.html#Learner.fit). After every epoch, all *metrics* will be printed and also made available to callbacks.
#
# The default weight decay will be `wd`, which will be handled using the method from [Fixing Weight Decay Regularization in Adam](https://arxiv.org/abs/1711.05101) if `true_wd` is set (otherwise it's L2 regularization). If `bn_wd` is `False`, then weight decay will be removed from batchnorm layers, as recommended in [Accurate, Large Minibatch SGD: Training ImageNet in 1 Hour](https://arxiv.org/abs/1706.02677). If `train_bn`, batchnorm layer learnable params are trained even for frozen layer groups.
#
# To use [discriminative layer training](#Discriminative-layer-training), pass a list of [`nn.Module`](https://pytorch.org/docs/stable/nn.html#torch.nn.Module) as `layer_groups`; each [`nn.Module`](https://pytorch.org/docs/stable/nn.html#torch.nn.Module) will be used to customize the optimization of the corresponding layer group.
#
# If `path` is provided, all the model files created will be saved in `path`/`model_dir`; if not, then they will be saved in `data.path`/`model_dir`.
#
# You can pass a list of [`callback`](/callback.html#callback)s that you have already created, or (more commonly) simply pass a list of callback functions to `callback_fns` and each function will be called (passing `self`) on object initialization, with the results stored as callback objects. For a walk-through, see the [training overview](/training.html) page. You may also want to use an [application](applications.html) specific model. For example, if you are dealing with a vision dataset, here the MNIST, you might want to use the [`cnn_learner`](/vision.learner.html#cnn_learner) method:

path = untar_data(URLs.MNIST_SAMPLE)
data = ImageDataBunch.from_folder(path)
learn = cnn_learner(data, models.resnet18, metrics=accuracy)


# ### Model fitting methods

show_doc(Learner.lr_find)


# Runs the learning rate finder defined in [`LRFinder`](/callbacks.lr_finder.html#LRFinder), as discussed in [Cyclical Learning Rates for Training Neural Networks](https://arxiv.org/abs/1506.01186).

learn.lr_find()


learn.recorder.plot()


show_doc(Learner.fit)


# Uses [discriminative layer training](#Discriminative-layer-training) if multiple learning rates or weight decay values are passed. To control training behaviour, use the [`callback`](/callback.html#callback) system or one or more of the pre-defined [`callbacks`](/callbacks.html#callbacks).

learn.fit(1)


show_doc(Learner.fit_one_cycle)


# Use cycle length `cyc_len`, a per cycle maximal learning rate `max_lr`, momentum `moms`, division factor `div_factor`, weight decay `wd`, and optional callbacks [`callbacks`](/callbacks.html#callbacks). Uses the [`OneCycleScheduler`](/callbacks.one_cycle.html#OneCycleScheduler) callback. Please refer to [What is 1-cycle](/callbacks.one_cycle.html#What-is-1cycle?) for a conceptual background of 1-cycle training policy and more technical details on what do the method's arguments do.

learn.fit_one_cycle(1)


# ### See results

show_doc(Learner.predict)


# `predict` can be used to get a single prediction from the trained learner on one specific piece of data you are interested in.

learn.data.train_ds[0]


# Each element of the dataset is a tuple, where the first element is the data itself, while the second element is the target label. So to get the data, we need to index one more time.

data = learn.data.train_ds[0][0]


data


pred = learn.predict(data)
pred


# The first two elements of the tuple are, respectively, the predicted class and label. Label here is essentially an internal representation of each class, since class name is a string and cannot be used in computation. To check what each label corresponds to, run:

learn.data.classes


# So category 0 is 3 while category 1 is 7.

probs = pred[2]


# The last element in the tuple is the predicted probabilities. For a categorization dataset, the number of probabilities returned is the same as the number of classes; `probs[i]` is the probability that the `item` belongs to `learn.data.classes[i]`.

learn.data.valid_ds[0][0]


# You could always check yourself if the probabilities given make sense.

show_doc(Learner.get_preds)


# It will run inference using the learner on all the data in the `ds_type` dataset and return the predictions; if `n_batch` is not specified, it will run the predictions on the default batch size. If `with_loss`, it will also return the loss on each prediction.

# Here is how you check the default batch size.

learn.data.batch_size


preds = learn.get_preds()
preds


# The first element of the tuple is a tensor that contains all the predictions.

preds[0]


# While the second element of the tuple is a tensor that contains all the target labels.

preds[1]


preds[1][0]


# For more details about what each number mean, refer to the documentation of [`predict`](/basic_train.html#predict).
#
# Since [`get_preds`](/basic_train.html#get_preds) gets predictions on all the data in the `ds_type` dataset, here the number of predictions will be equal to the number of data in the validation dataset.

len(learn.data.valid_ds)


len(preds[0]), len(preds[1])


# To get predictions on the entire training dataset, simply set the `ds_type` argument accordingly.

learn.get_preds(ds_type=DatasetType.Train)


# To also get prediction loss along with the predictions and the targets, set `with_loss=True` in the arguments.

learn.get_preds(with_loss=True)


# Note that the third tensor in the output tuple contains the losses.

show_doc(Learner.validate)


# Return the calculated loss and the metrics of the current model on the given data loader `dl`. The default data loader `dl` is the validation dataloader.

# You can check the default metrics of the learner using:

str(learn.metrics)


learn.validate()


learn.validate(learn.data.valid_dl)


learn.validate(learn.data.train_dl)


show_doc(Learner.show_results)


# Note that the text number on the top is the ground truth, or the target label, the one in the middle is the prediction, while the image number on the bottom is the image data itself.

learn.show_results()


learn.show_results(ds_type=DatasetType.Train)


show_doc(Learner.pred_batch)


# Note that the number of predictions given equals to the batch size.

learn.data.batch_size


preds = learn.pred_batch()
len(preds)


# Since the total number of predictions is too large, we will only look at a part of them.

preds[:10]


item = learn.data.train_ds[0][0]
item


batch = learn.data.one_item(item)
batch


learn.pred_batch(batch=batch)


show_doc(Learner.interpret, full_name='interpret')


jekyll_note('This function only works in the vision application.')


# For more details, refer to [ClassificationInterpretation](/vision.learner.html#ClassificationInterpretation)

# ### Model summary

show_doc(Learner.summary)


# ### Test time augmentation

show_doc(Learner.TTA, full_name='TTA')


# Applies Test Time Augmentation to `learn` on the dataset `ds_type`. We take the average of our regular predictions (with a weight `beta`) with the average of predictions obtained through augmented versions of the training set (with a weight `1-beta`). The transforms decided for the training set are applied with a few changes `scale` controls the scale for zoom (which isn't random), the cropping isn't random but we make sure to get the four corners of the image. Flipping isn't random but applied once on each of those corner images (so that makes 8 augmented versions total).

# ### Gradient clipping

show_doc(Learner.clip_grad)


# ### Mixed precision training

show_doc(Learner.to_fp16)


# Uses the [`MixedPrecision`](/callbacks.fp16.html#MixedPrecision) callback to train in mixed precision (i.e. forward and backward passes using fp16, with weight updates using fp32), using all [NVIDIA recommendations](https://docs.nvidia.com/deeplearning/sdk/mixed-precision-training/index.html) for ensuring speed and accuracy.

show_doc(Learner.to_fp32)


# ### Distributed training

# If you want to use ditributed training or [`torch.nn.DataParallel`](https://pytorch.org/docs/stable/nn.html#torch.nn.DataParallel) these will directly wrap the model for you.

show_doc(Learner.to_distributed, full_name='to_distributed')


show_doc(Learner.to_parallel, full_name='to_parallel')


# ### Discriminative layer training

# When fitting a model you can pass a list of learning rates (and/or weight decay amounts), which will apply a different rate to each *layer group* (i.e. the parameters of each module in `self.layer_groups`). See the [Universal Language Model Fine-tuning for Text Classification](https://arxiv.org/abs/1801.06146) paper for details and experimental results in NLP (we also frequently use them successfully in computer vision, but have not published a paper on this topic yet). When working with a [`Learner`](/basic_train.html#Learner) on which you've called `split`, you can set hyperparameters in four ways:
#
# 1. `param = [val1, val2 ..., valn]` (n = number of layer groups)
# 2. `param = val`
# 3. `param = slice(start,end)`
# 4. `param = slice(end)`
#
# If we chose to set it in way 1, we must specify a number of values exactly equal to the number of layer groups. If we chose to set it in way 2, the chosen value will be repeated for all layer groups. See [`Learner.lr_range`](/basic_train.html#Learner.lr_range) for an explanation of the `slice` syntax).
#
# Here's an example of how to use discriminative learning rates (note that you don't actually need to manually call [`Learner.split`](/basic_train.html#Learner.split) in this case, since fastai uses this exact function as the default split for `resnet18`; this is just to show how to customize it):

# creates 3 layer groups
learn.split(lambda m: (m[0][6], m[1]))
# only randomly initialized head now trainable
learn.freeze()


learn.fit_one_cycle(1)


# all layers now trainable
learn.unfreeze()
# optionally, separate LR and WD for each group
learn.fit_one_cycle(1, max_lr=(1e-4, 1e-3, 1e-2), wd=(1e-4, 1e-4, 1e-1))


show_doc(Learner.lr_range)


# Rather than manually setting an LR for every group, it's often easier to use [`Learner.lr_range`](/basic_train.html#Learner.lr_range). This is a convenience method that returns one learning rate for each layer group. If you pass `slice(start,end)` then the first group's learning rate is `start`, the last is `end`, and the remaining are evenly geometrically spaced.
#
# If you pass just `slice(end)` then the last group's learning rate is `end`, and all the other groups are `end/10`. For instance (for our learner that has 3 layer groups):

learn.lr_range(slice(1e-5, 1e-3)), learn.lr_range(slice(1e-3))


show_doc(Learner.unfreeze)


# Sets every layer group to *trainable* (i.e. `requires_grad=True`).

show_doc(Learner.freeze)


# Sets every layer group except the last to *untrainable* (i.e. `requires_grad=False`).
#
# What does '**the last layer group**' mean?
#
# In the case of transfer learning, such as `learn = cnn_learner(data, models.resnet18, metrics=error_rate)`, `learn.model`will print out two large groups of layers: (0) Sequential and (1) Sequental in the following structure. We can consider the last conv layer as the break line between the two groups.
# ```
# Sequential(
#   (0): Sequential(
#     (0): Conv2d(3, 64, kernel_size=(7, 7), stride=(2, 2), padding=(3, 3), bias=False)
#     (1): BatchNorm2d(64, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
#     (2): ReLU(inplace)
#     ...
#
#             (conv2): Conv2d(512, 512, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False)
#              (bn2): BatchNorm2d(512, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
#       )
#     )
#   )
#   (1): Sequential(
#     (0): AdaptiveConcatPool2d(
#       (ap): AdaptiveAvgPool2d(output_size=1)
#       (mp): AdaptiveMaxPool2d(output_size=1)
#     )
#     (1): Flatten()
#     (2): BatchNorm1d(1024, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
#     (3): Dropout(p=0.25)
#     (4): Linear(in_features=1024, out_features=512, bias=True)
#     (5): ReLU(inplace)
#     (6): BatchNorm1d(512, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
#     (7): Dropout(p=0.5)
#     (8): Linear(in_features=512, out_features=12, bias=True)
#   )
# )
# ```
#
# `learn.freeze` freezes the first group and keeps the second or last group free to train, including multiple layers inside (this is why calling it 'group'), as you can see in `learn.summary()` output. How to read the table below, please see [model summary docs](/callbacks.hooks.html#model_summary).
#
# ```
# ======================================================================
# Layer (type)         Output Shape         Param #    Trainable
# ======================================================================
# ...
# ...
# ...
# ______________________________________________________________________
# Conv2d               [1, 512, 4, 4]       2,359,296  False
# ______________________________________________________________________
# BatchNorm2d          [1, 512, 4, 4]       1,024      True
# ______________________________________________________________________
# AdaptiveAvgPool2d    [1, 512, 1, 1]       0          False
# ______________________________________________________________________
# AdaptiveMaxPool2d    [1, 512, 1, 1]       0          False
# ______________________________________________________________________
# Flatten              [1, 1024]            0          False
# ______________________________________________________________________
# BatchNorm1d          [1, 1024]            2,048      True
# ______________________________________________________________________
# Dropout              [1, 1024]            0          False
# ______________________________________________________________________
# Linear               [1, 512]             524,800    True
# ______________________________________________________________________
# ReLU                 [1, 512]             0          False
# ______________________________________________________________________
# BatchNorm1d          [1, 512]             1,024      True
# ______________________________________________________________________
# Dropout              [1, 512]             0          False
# ______________________________________________________________________
# Linear               [1, 12]              6,156      True
# ______________________________________________________________________
#
# Total params: 11,710,540
# Total trainable params: 543,628
# Total non-trainable params: 11,166,912
# ```
#

show_doc(Learner.freeze_to)


# From above we know what is layer group, but **what exactly does `freeze_to` do behind the scenes**?
#
# The `freeze_to` source code can be understood as the following pseudo-code:
# ```python
# def freeze_to(self, n:int)->None:
#     for g in self.layer_groups[:n]: freeze
#     for g in self.layer_groups[n:]: unfreeze
# ```
# In other words, for example, `freeze_to(1)` is to freeze layer group 0 and unfreeze the rest layer groups, and `freeze_to(3)` is to freeze layer groups 0, 1, and 2 but unfreeze the rest layer groups (if there are more layer groups left).
#
# Both `freeze` and `unfreeze` [sources](https://github.com/fastai/fastai/blob/master/fastai/basic_train.py#L216) are defined using `freeze_to`:
# - When we say `freeze`, we mean that in the specified layer groups the [`requires_grad`](/torch_core.html#requires_grad) of all layers with weights (except BatchNorm layers) are set `False`, so the layer weights won't be updated during training.
# - when we say `unfreeze`, we mean that in the specified layer groups the [`requires_grad`](/torch_core.html#requires_grad) of all layers with weights (except BatchNorm layers) are set `True`, so the layer weights will be updated during training.

show_doc(Learner.split)


# A convenience method that sets `layer_groups` based on the result of [`split_model`](/torch_core.html#split_model). If `split_on` is a function, it calls that function and passes the result to [`split_model`](/torch_core.html#split_model) (see above for example).

# ### Saving and loading models

# Simply call [`Learner.save`](/basic_train.html#Learner.save) and [`Learner.load`](/basic_train.html#Learner.load) to save and load models. Only the parameters are saved, not the actual architecture (so you'll need to create your model in the same way before loading weights back in). Models are saved to the `path`/`model_dir` directory.

show_doc(Learner.save)


# If agument `name` is a pathlib object that's an absolute path, it'll override the default base directory (`learn.path`), otherwise the model will be saved in a file relative to `learn.path`.

learn.save("trained_model")


learn.save("trained_model", return_path=True)


show_doc(Learner.load)


# This method only works after `save` (don't confuse with `export`/[`load_learner`](/basic_train.html#load_learner) pair).
#
# If the `purge` argument is `True` (default) `load` internally calls `purge` with `clear_opt=False` to presever `learn.opt`.

learn = learn.load("trained_model")


# ### Deploying your model

# When you are ready to put your model in production, export the minimal state of your [`Learner`](/basic_train.html#Learner) with:

show_doc(Learner.export)


# If agument `fname` is a pathlib object that's an absolute path, it'll override the default base directory (`learn.path`), otherwise the model will be saved in a file relative to `learn.path`.

# Passing `destroy=True` will destroy the [`Learner`](/basic_train.html#Learner), freeing most of its memory consumption. For specifics see [`Learner.destroy`](/basic_train.html#Learner.destroy).
#
# This method only works with the [`Learner`](/basic_train.html#Learner) whose [`data`](/vision.data.html#vision.data) was created through the [data block API](/data_block.html).
#
# Otherwise, you will have to create a [`Learner`](/basic_train.html#Learner) yourself at inference and load the model with [`Learner.load`](/basic_train.html#Learner.load).

learn.export()


learn.export('trained_model.pkl')


path = learn.path
path


show_doc(load_learner)


# This function only works after `export` (don't confuse with `save`/`load` pair).
#
# The `db_kwargs` will be passed to the call to `databunch` so you can specify a `bs` for the test set, or `num_workers`.

learn = load_learner(path)


learn = load_learner(path, 'trained_model.pkl')


# WARNING: If you used any customized classes when creating your learner, you must first define these classes first before executing [`load_learner`](/basic_train.html#load_learner).
#
# You can find more information and multiple examples in [this tutorial](/tutorial.inference.html).

# ### Freeing memory
#
# If you want to be able to do more without needing to restart your notebook, the following methods are designed to free memory when it's no longer needed.
#
# Refer to [this tutorial](/tutorial.resources.html) to learn how and when to use these methods.

show_doc(Learner.purge)


# If `learn.path` is read-only, you can set `model_dir` attribute in Learner to a full `libpath` path that is writable (by setting `learn.model_dir` or passing `model_dir` argument in the [`Learner`](/basic_train.html#Learner) constructor).

show_doc(Learner.destroy)


# If you need to free the memory consumed by the [`Learner`](/basic_train.html#Learner) object, call this method.
#
# It can also be automatically invoked through [`Learner.export`](/basic_train.html#Learner.export) via its `destroy=True` argument.

# ### Other methods

show_doc(Learner.init)


# Initializes all weights (except batchnorm) using function `init`, which will often be from PyTorch's [`nn.init`](https://pytorch.org/docs/stable/nn.html#torch-nn-init) module.

show_doc(Learner.mixup)


# Uses [`MixUpCallback`](/callbacks.mixup.html#MixUpCallback).

show_doc(Learner.backward)


show_doc(Learner.create_opt)


# You generally won't need to call this yourself - it's used to create the [`optim`](https://pytorch.org/docs/stable/optim.html#module-torch.optim) optimizer before fitting the model.

show_doc(Learner.dl)


learn.dl()


learn.dl(DatasetType.Train)


show_doc(Recorder, title_level=2)


# A [`Learner`](/basic_train.html#Learner) creates a [`Recorder`](/basic_train.html#Recorder) object automatically - you do not need to explicitly pass it to `callback_fns` - because other callbacks rely on it being available. It stores the smoothed loss, hyperparameter values, and metrics for each batch, and provides plotting methods for each. Note that [`Learner`](/basic_train.html#Learner) automatically sets an attribute with the snake-cased name of each callback, so you can access this through `Learner.recorder`, as shown below.

# ### Plotting methods

show_doc(Recorder.plot)


# This is mainly used with the learning rate finder, since it shows a scatterplot of loss vs learning rate.

path = untar_data(URLs.MNIST_SAMPLE)
data = ImageDataBunch.from_folder(path)
learn = cnn_learner(data, models.resnet18, metrics=accuracy)


learn.lr_find()
learn.recorder.plot()


show_doc(Recorder.plot_losses)


# Note that validation losses are only calculated once per epoch, whereas training losses are calculated after every batch.

learn.fit_one_cycle(5)
learn.recorder.plot_losses()


show_doc(Recorder.plot_lr)


learn.recorder.plot_lr()


learn.recorder.plot_lr(show_moms=True)


show_doc(Recorder.plot_metrics)


# Note that metrics are only collected at the end of each epoch, so you'll need to train at least two epochs to have anything to show here.

learn.recorder.plot_metrics()


# ### Callback methods

# You don't call these yourself - they're called by fastai's [`Callback`](/callback.html#Callback) system automatically to enable the class's functionality. Refer to [`Callback`](/callback.html#Callback) for more details.

show_doc(Recorder.on_backward_begin)


show_doc(Recorder.on_batch_begin)


show_doc(Recorder.on_epoch_end)


show_doc(Recorder.on_train_begin)


# ### Inner functions

# The following functions are used along the way by the [`Recorder`](/basic_train.html#Recorder) or can be called by other callbacks.

show_doc(Recorder.add_metric_names)


show_doc(Recorder.format_stats)


# ## Module functions

# Generally you'll want to use a [`Learner`](/basic_train.html#Learner) to train your model, since they provide a lot of functionality and make things easier. However, for ultimate flexibility, you can call the same underlying functions that [`Learner`](/basic_train.html#Learner) calls behind the scenes:

show_doc(fit)


# Note that you have to create the [`Optimizer`](https://pytorch.org/docs/stable/optim.html#torch.optim.Optimizer) yourself if you call this function, whereas [`Learn.fit`](/basic_train.html#fit) creates it for you automatically.

show_doc(train_epoch)


# You won't generally need to call this yourself - it's what [`fit`](/basic_train.html#fit) calls for each epoch.

show_doc(validate)


# This is what [`fit`](/basic_train.html#fit) calls after each epoch. You can call it if you want to run inference on a [`DataLoader`](https://pytorch.org/docs/stable/data.html#torch.utils.data.DataLoader) manually.

show_doc(get_preds)


show_doc(loss_batch)


# You won't generally need to call this yourself - it's what [`fit`](/basic_train.html#fit) and [`validate`](/basic_train.html#validate) call for each batch. It only does a backward pass if you set `opt`.

# ## Other classes

show_doc(LearnerCallback, title_level=3)


show_doc(RecordOnCPU, title_level=3)


# ## Undocumented Methods - Methods moved below this line will intentionally be hidden

show_doc(Learner.tta_only)


show_doc(Learner.TTA)


show_doc(RecordOnCPU.on_batch_begin)


# ## New Methods - Please document or move to the undocumented section
