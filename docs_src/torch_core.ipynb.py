
# coding: utf-8

# ## Torch Core

# This module contains all the basic functions we need in other modules of the fastai library (split with [`core`](/core.html#core) that contains the ones not requiring pytorch). Its documentation can easily be skipped at a first read, unless you want to know what a given function does.

from fastai.imports import *
from fastai.gen_doc.nbdoc import *
from fastai.layers import *
from fastai.torch_core import *


# ## Global constants

# `AdamW = partial(optim.Adam, betas=(0.9,0.99))` <div style="text-align: right"><a href="https://github.com/fastai/fastai/blob/master/fastai/torch_core.py#L43">[source]</a></div>

# `bn_types = (nn.BatchNorm1d, nn.BatchNorm2d, nn.BatchNorm3d)` <div style="text-align: right"><a href="https://github.com/fastai/fastai/blob/master/fastai/torch_core.py#L41">[source]</a></div>

# `defaults.device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')` <div style="text-align: right"><a href="https://github.com/fastai/fastai/blob/master/fastai/torch_core.py#L62">[source]</a></div>

# If you are trying to make fastai run on the CPU, simply change the default device: `defaults.device = 'cpu'`.
#
# Alternatively, if not using wildcard imports: `fastai.torch_core.defaults.device = 'cpu'`.

# ## Functions that operate conversions

show_doc(batch_to_half)


show_doc(flatten_model, full_name='flatten_model')


# Flattens all the layers of `m` into an array. This allows for easy access to the layers of the model and allows you to manipulate the model as if it was an array.

m = simple_cnn([3, 6, 12])
m


flatten_model(m)


show_doc(model2half)


# Converting model parameters to half precision allows us to leverage fast `FP16` arithmetic which can speed up the computations by 2-8 times. It also reduces memory consumption allowing us to train deeper models.
#
# **Note**: Batchnorm layers are not converted to half precision as that may lead to instability in training.

m = simple_cnn([3, 6, 12], bn=True)

def show_params_dtype(state_dict):
    """Simple function to pretty print the dtype of the model params"""
    for wt_name, param in state_dict.items():
        print("{:<30}: {}".format(wt_name, str(param.dtype)))
    print()

print("dtypes of model parameters before model2half: ")
show_params_dtype(m.state_dict())

# Converting model to half precision
m_half = model2half(m)

print("dtypes of model parameters after model2half: ")
show_params_dtype(m_half.state_dict())


show_doc(np2model_tensor)


# It is a wrapper on top of Pytorch's `torch.as_tensor` which converts numpy array to torch tensor, and additionally attempts to map all floats to `torch.float32` and all integers to `torch.int64` for consistencies in model data. Below is an example demonstrating it's functionality for floating number, similar functionality applies to integer as well.

a1 = np.ones((2, 3)).astype(np.float16)
a2 = np.ones((2, 3)).astype(np.float32)
a3 = np.ones((2, 3)).astype(np.float64)

b1 = np2model_tensor(a1) # Maps to torch.float32
b2 = np2model_tensor(a2) # Maps to torch.float32
b3 = np2model_tensor(a3) # Maps to torch.float32

print(f"Datatype of as': {a1.dtype}, {a2.dtype}, {a3.dtype}")
print(f"Datatype of bs': {b1.dtype}, {b2.dtype}, {b3.dtype}")


show_doc(requires_grad)


# Performs both getting and setting of [`requires_grad`](/torch_core.html#requires_grad) parameter of the tensors, which decided whether to accumulate gradients or not.
#
# * If `b` is `None`: The function **gets** the [`requires_grad`](/torch_core.html#requires_grad) for the model parameter, to be more specific it returns the [`requires_grad`](/torch_core.html#requires_grad) of the first element in the model.
#
# * Else if `b` is passed (a boolean value), [`requires_grad`](/torch_core.html#requires_grad) of all parameters of the model is **set** to `b`.

# Any Pytorch model
m = simple_cnn([3, 6, 12], bn=True)

# Get the requires_grad of model
print("requires_grad of model: {}".format(requires_grad(m)))

# Set requires_grad of all params in model to false
requires_grad(m, False)

# Get the requires_grad of model
print("requires_grad of model: {}".format(requires_grad(m)))


show_doc(tensor)


# Handy function when you want to convert any list type object to tensor, initialize your weights manually, and other similar cases.
#
# **NB**: When passing multiple vectors, all vectors must be of same dimensions. (Obvious but can be forgotten sometimes)

# Conversion from any numpy array
b = tensor(np.array([1, 2, 3]))
print(b, type(b))

# Passing as multiple parameters
b = tensor(1, 2, 3)
print(b, type(b))

# Passing a single list
b = tensor([1, 2, 3])
print(b, type(b))

# Can work with multiple vectors / lists
b = tensor([1, 2], [3, 4])
print(b, type(b))


show_doc(to_cpu)


# A wrapper on top of Pytorch's `torch.Tensor.cpu()` function, which creates and returns a copy of a tensor or even a **list** of tensors in the CPU. As described in Pytorch's docs, if the tensor or list of tensor is already on the CPU, the exact data is returned and no copy is made.
#
# Useful to convert all the list of parameters of the model to CPU in a single call.

if torch.cuda.is_available():
    a = [torch.randn((1, 1)).cuda() for i in range(3)]
    print(a)
    print("Id of tensors in a: ")
    for i in a: print(id(i))
    
    # Getting a CPU version of the tensors in GPU
    b = to_cpu(a)
    print(b)
    print("Id of tensors in b:")
    for i in b: print(id(i))
    
    # Trying to perform to_cpu on a list of tensor already in CPU
    c = to_cpu(b)
    print(c)
    # The tensors in c has exact id as that of b. No copy performed.
    print("Id of tensors in c:")
    for i in c: print(id(i))


show_doc(to_data)


# Returns the data attribute from the object or collection of objects that inherits from [`ItemBase`](/core.html#ItemBase) class. Useful to examine the exact values of the data, could be used to work with the data outside of `fastai` classes.

# Default example examined

from fastai import *
from fastai.vision import *

path = untar_data(URLs.MNIST_SAMPLE)
data = ImageDataBunch.from_folder(path)

# Examin the labels
ys = list(data.y)
print("Category display names: ", [ys[0], ys[-1]])

print("Unique classes internally represented as: ", to_data([ys[0], ys[-1]]))


show_doc(to_detach)


show_doc(to_device)


show_doc(to_half)


# Converts the tensor or list of `FP16`, resulting in less memory consumption and faster computations with the tensor. It does not convert `torch.int` types to half precision.

a1 = torch.tensor([1, 2], dtype=torch.int64)
a2 = torch.tensor([1, 2], dtype=torch.int32)
a3 = torch.tensor([1, 2], dtype=torch.int16)
a4 = torch.tensor([1, 2], dtype=torch.float64)
a5 = torch.tensor([1, 2], dtype=torch.float32)
a6 = torch.tensor([1, 2], dtype=torch.float16)

print("dtype of as: ", a1.dtype, a2.dtype, a3.dtype, a4.dtype, a5.dtype, a6.dtype, sep="\t")

b1, b2, b3, b4, b5, b6 = to_half([a1, a2, a3, a4, a5, a6])

print("dtype of bs: ", b1.dtype, b2.dtype, b3.dtype, b4.dtype, b5.dtype, b6.dtype, sep="\t")


show_doc(to_np)


# Internally puts the data to CPU, and converts to `numpy.ndarray` equivalent of `torch.tensor` by calling `torch.Tensor.numpy()`.

a = torch.tensor([1, 2], dtype=torch.float64)

if torch.cuda.is_available():
    a = a.cuda()

print(a, type(a), a.device)

b = to_np(a)

print(b, type(b))


show_doc(try_int)


# Converts floating point numbers to integer
print(try_int(12.5), type(try_int(12.5)))

# This is a Rank-1 ndarray, which ideally should not be converted to int
print(try_int(np.array([1.5])), try_int(np.array([1.5])).dtype)

# Numpy array with a single elements are converted to int
print(try_int(np.array(1.5)), type(try_int(np.array(1.5))))

print(try_int(torch.tensor(2.5)), type(try_int(torch.tensor(2.5))))

# Strings are not converted to int (of course)
print(try_int("12.5"), type(try_int("12.5")))


# ## Functions to deal with model initialization

show_doc(apply_init)


show_doc(apply_leaf)


show_doc(cond_init)


show_doc(in_channels)


show_doc(init_default)


# ## Functions to get information of a model

show_doc(children)


show_doc(children_and_parameters)


show_doc(first_layer)


show_doc(last_layer)


show_doc(num_children)


show_doc(one_param)


show_doc(range_children)


show_doc(trainable_params)


# ## Functions to deal with BatchNorm layers

show_doc(bn2float)


show_doc(set_bn_eval)


show_doc(split_no_wd_params)


# This is used by the optimizer to determine which params should be applied weight decay when using the option `bn_wd=False` is used in a [`Learner`](/basic_train.html#Learner).

# ## Functions to get random tensors

show_doc(log_uniform)


log_uniform(0.5, 2, (8,))


show_doc(rand_bool)


rand_bool(0.5, 8)


show_doc(uniform)


uniform(0, 1, (8,))


show_doc(uniform_int)


uniform_int(0, 2, (8,))


# ## Other functions

show_doc(ModelOnCPU, title_level=3)


show_doc(NoneReduceOnCPU, title_level=3)


show_doc(ParameterModule, title_level=3)


show_doc(data_collate)


show_doc(get_model)


show_doc(grab_idx)


show_doc(logit)


show_doc(logit_)


show_doc(model_type)


show_doc(np_address)


show_doc(split_model)


# If `splits` are layers, the model is split at those (not included) sequentially. If `want_idxs` is True, the corresponding indexes are returned. If `splits` are lists of layers, the model is split according to those.

show_doc(split_model_idx)


show_doc(trange_of)


# ## Undocumented Methods - Methods moved below this line will intentionally be hidden

show_doc(tensor__array__)


show_doc(ParameterModule.forward)


# ## New Methods - Please document or move to the undocumented section

show_doc(to_float)


show_doc(flatten_check)
