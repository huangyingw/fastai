
# coding: utf-8

# ## IPython Utilities

# Utilities to help work with ipython/jupyter environment.

# To import from [`fastai.utils.ipython`](/utils.ipython.html#utils.ipython) do:

from fastai.gen_doc.nbdoc import *


from fastai.utils.ipython import *


# ## Workarounds to the leaky ipython traceback on exception
#
# ipython has a feature where it stores tb with all the `locals()` tied in, which
# prevents `gc.collect()` from freeing those variables and leading to a leakage.
#
# Therefore we cleanse the tb before handing it over to ipython. The 2 ways of doing it are by either using the [`gpu_mem_restore`](/utils.ipython.html#gpu_mem_restore) decorator or the [`gpu_mem_restore_ctx`](/utils.ipython.html#gpu_mem_restore_ctx) context manager which are described next:

show_doc(gpu_mem_restore)


# [`gpu_mem_restore`](/utils.ipython.html#gpu_mem_restore) is a decorator to be used with any functions that interact with CUDA (top-level is fine)
#
# * under non-ipython environment it doesn't do anything.
# * under ipython currently it strips tb by default only for the "CUDA out of memory" exception.
#
# The env var `FASTAI_TB_CLEAR_FRAMES` changes this behavior when run under ipython,
# depending on its value:
#
# * "0": never  strip tb (makes it possible to always use `%debug` magic, but with leaks)
# * "1": always strip tb (never need to worry about leaks, but `%debug` won't work)
#
# e.g. `os.environ['FASTAI_TB_CLEAR_FRAMES']="0"` will set it to 0.
#

show_doc(gpu_mem_restore_ctx, title_level=4)


# if function decorator is not a good option, you can use a context manager instead. For example:
# ```
# with gpu_mem_restore_ctx():
#    learn.fit_one_cycle(1,1e-2)
# ```
# This particular one will clear tb on any exception.

from fastai.gen_doc.nbdoc import *
from fastai.utils.ipython import *


# ## Undocumented Methods - Methods moved below this line will intentionally be hidden
