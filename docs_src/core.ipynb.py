
# coding: utf-8

# ## Basic core

# This module contains all the basic functions we need in other modules of the fastai library (split with [`torch_core`](/torch_core.html#torch_core) that contains the ones requiring pytorch). Its documentation can easily be skipped at a first read, unless you want to know what a given function does.

from fastai.gen_doc.nbdoc import *
from fastai.core import *


# ## Global constants

# `default_cpus = min(16, num_cpus())` <div style="text-align: right"><a href="https://github.com/fastai/fastai/blob/master/fastai/core.py#L45">[source]</a></div>

# ## Check functions

show_doc(has_arg)


# Examples for two [`fastai.core`](/core.html#core) functions.  Docstring shown before calling [`has_arg`](/core.html#has_arg) for reference
#

has_arg(download_url, 'url')


has_arg(index_row, 'x')


has_arg(index_row, 'a')


show_doc(ifnone)


param, alt_param = None, 5
ifnone(param, alt_param)


param, alt_param = None, [1, 2, 3]
ifnone(param, alt_param)


show_doc(is1d)


two_d_array = np.arange(12).reshape(6, 2)
print(two_d_array)
print(is1d(two_d_array))


is1d(two_d_array.flatten())


show_doc(is_listy)


# Check if `x` is a `Collection`. `Tuple` or `List` qualify

some_data = [1, 2, 3]
is_listy(some_data)


some_data = (1, 2, 3)
is_listy(some_data)


some_data = 1024
print(is_listy(some_data))


print(is_listy([some_data]))


some_data = dict([('a', 1), ('b', 2), ('c', 3)])
print(some_data)
print(some_data.keys())


print(is_listy(some_data))
print(is_listy(some_data.keys()))


print(is_listy(list(some_data.keys())))


show_doc(is_tuple)


# Check if `x` is a `tuple`.

print(is_tuple([1, 2, 3]))


print(is_tuple((1, 2, 3)))


# ## Collection related functions

show_doc(arange_of)


arange_of([5, 6, 7])


type(arange_of([5, 6, 7]))


show_doc(array)


array([1, 2, 3])


# Note that after we call the generator, we do not reset.  So the [`array`](/core.html#array) call has 5 less entries than it would if we ran from the start of the generator.

def data_gen():
    i = 100.01
    while i < 200:
        yield i
        i += 1.

ex_data_gen = data_gen()
for _ in range(5):
    print(next(ex_data_gen))


array(ex_data_gen)


ex_data_gen_int = data_gen()

array(ex_data_gen_int, dtype=int)  #Cast output to int array


show_doc(arrays_split)


data_a = np.arange(15)
data_b = np.arange(15)[::-1]

mask_a = (data_a > 10)
print(data_a)
print(data_b)
print(mask_a)


arrays_split(mask_a, data_a)


np.vstack([data_a, data_b]).transpose().shape


arrays_split(mask_a, np.vstack([data_a, data_b]).transpose()) #must match on dimension 0


show_doc(chunks)


# You can transform a `Collection` into an `Iterable` of 'n' sized chunks by calling [`chunks`](/core.html#chunks):

data = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9]
for chunk in chunks(data, 2):
    print(chunk)


for chunk in chunks(data, 3):
    print(chunk)


show_doc(df_names_to_idx)


ex_df = pd.DataFrame.from_dict({"a": [1, 1, 1], "b": [2, 2, 2]})
print(ex_df)


df_names_to_idx('b', ex_df)


show_doc(extract_kwargs)


key_word_args = {"a": 2, "some_list": [1, 2, 3], "param": 'mean'}
key_word_args


(extracted_val, remainder) = extract_kwargs(['param'], key_word_args)
print(extracted_val, remainder)


show_doc(idx_dict)


idx_dict(['a', 'b', 'c'])


show_doc(index_row)


data = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9]
index_row(data, 4)


index_row(pd.Series(data), 7)


data_df = pd.DataFrame([data[::-1], data]).transpose()
data_df


index_row(data_df, 7)


show_doc(listify)


to_match = np.arange(12)
listify('a', to_match)


listify('a', 5)


listify(77.1, 3)


listify((1, 2, 3))


listify((1, 2, 3), ('a', 'b', 'c'))


show_doc(random_split)


# Splitting is done here with `random.uniform()` so you may not get the exact split percentage for small data sets

data = np.arange(20).reshape(10, 2)
data.tolist()


random_split(0.20, data.tolist())


random_split(0.20, pd.DataFrame(data))


show_doc(range_of)


range_of([5, 4, 3])


range_of(np.arange(10)[::-1])


show_doc(series2cat)


data_df = pd.DataFrame.from_dict({"a": [1, 1, 1, 2, 2, 2], "b": ['f', 'e', 'f', 'g', 'g', 'g']})
data_df


data_df['b']


series2cat(data_df, 'b')
data_df['b']


series2cat(data_df, 'a')
data_df['a']


show_doc(split_kwargs_by_func)


key_word_args = {'url': 'http://fast.ai', 'dest': './', 'new_var': [1, 2, 3], 'testvalue': 42}
split_kwargs_by_func(key_word_args, download_url)


show_doc(to_int)


to_int(3.1415)


data = [1.2, 3.4, 7.25]
to_int(data)


show_doc(uniqueify)


uniqueify(pd.Series(data=['a', 'a', 'b', 'b', 'f', 'g']))


# ## Files management and downloads

show_doc(download_url)


show_doc(find_classes)


show_doc(join_path)


show_doc(join_paths)


show_doc(loadtxt_str)


show_doc(save_texts)


# ## Multiprocessing

show_doc(num_cpus)


show_doc(parallel)


# `func` must accept both the value and index of each `arr` element.

def my_func(value, index):
    print("Index: {}, Value: {}".format(index, value))
 
my_array = [i * 2 for i in range(5)]
parallel(my_func, my_array, max_workers=3)


show_doc(partition)


show_doc(partition_by_cores)


# ## Data block API

show_doc(ItemBase, title_level=3)


# All items used in fastai should subclass this. Must have a [`data`](/tabular.data.html#tabular.data) field that will be used when collating in mini-batches.

show_doc(ItemBase.apply_tfms)


show_doc(ItemBase.show)


# The default behavior is to set the string representation of this object as title of `ax`.

show_doc(Category, title_level=3)


# Create a [`Category`](/core.html#Category) with an `obj` of index [`data`](/tabular.data.html#tabular.data) in a certain classes list.

show_doc(EmptyLabel, title_level=3)


show_doc(MultiCategory, title_level=3)


# Create a [`MultiCategory`](/core.html#MultiCategory) with an `obj` that is a collection of labels. [`data`](/tabular.data.html#tabular.data) corresponds to the one-hot encoded labels and `raw` is a list of associated string.

show_doc(FloatItem)


# ## Others

show_doc(camel2snake)


camel2snake('DeviceDataLoader')


show_doc(even_mults)


# In linear scales each element is equidistant from its neighbors:

# from 1 to 10 in 5 steps
t = np.linspace(1, 10, 5)
t


for i in range(len(t) - 1):
    print(t[i + 1] - t[i])


# In logarithmic scales, each element is a multiple of the previous entry:

t = even_mults(1, 10, 5)
t


# notice how each number is a multiple of its predecessor
for i in range(len(t) - 1):
    print(t[i + 1] / t[i])


show_doc(func_args)


func_args(download_url)


# Additionally, [`func_args`](/core.html#func_args) can be used with functions that do not belong to the fastai library

func_args(np.linspace)


show_doc(noop)


# Return `x`.

# object is returned as-is
noop([1, 2, 3])


show_doc(one_hot)


# One-hot encoding is a standard machine learning technique. Assume we are dealing with a 10-class classification problem and we are supplied a list of labels:

y = [1, 4, 4, 5, 7, 9, 2, 4, 0]


jekyll_note("""y is zero-indexed, therefore its first element (1) belongs to class 2, its second element (4) to class 5 and so on.""")


len(y)


# y can equivalently be expressed as a matrix of 9 rows and 10 columns, where each row represents one element of the original y.

for label in y:
    print(one_hot(label, 10))


show_doc(show_some)


# select 3 elements from a list
some_data = show_some([10, 20, 30, 40, 50], 3)
some_data


type(some_data)


# the separator can be changed
some_data = show_some([10, 20, 30, 40, 50], 3, sep='---')
some_data


some_data[:-3]


# [`show_some`](/core.html#show_some) can take as input any class with \_\_len\_\_ and \_\_getitem\_\_

class Any(object):
    def __init__(self, data):
        self.data = data

    def __len__(self):
        return len(self.data)

    def __getitem__(self, i):
        return self.data[i]
 
some_other_data = Any('nice')
show_some(some_other_data, 2)


show_doc(subplots)


show_doc(text2html_table)


# ## Undocumented Methods - Methods moved below this line will intentionally be hidden

# ## New Methods - Please document or move to the undocumented section

show_doc(is_dict)
