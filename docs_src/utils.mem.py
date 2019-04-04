
# coding: utf-8

# ## Memory management utils

# Utility functions for memory management. Currently primarily for GPU.

from fastai.gen_doc.nbdoc import *
from fastai.utils.mem import *


show_doc(gpu_mem_get)


# [`gpu_mem_get`](/utils.mem.html#gpu_mem_get)
#
# * for gpu returns `GPUMemory(total, free, used)`
# * for cpu returns `GPUMemory(0, 0, 0)`
# * for invalid gpu id returns `GPUMemory(0, 0, 0)`

show_doc(gpu_mem_get_all)


# [`gpu_mem_get_all`](/utils.mem.html#gpu_mem_get_all)
# * for gpu returns `[ GPUMemory(total_0, free_0, used_0), GPUMemory(total_1, free_1, used_1), .... ]`
# * for cpu returns `[]`
#

show_doc(gpu_mem_get_free)


show_doc(gpu_mem_get_free_no_cache)


show_doc(gpu_mem_get_used)


show_doc(gpu_mem_get_used_no_cache)


# [`gpu_mem_get_used_no_cache`](/utils.mem.html#gpu_mem_get_used_no_cache)

show_doc(gpu_mem_get_used_fast)


# [`gpu_mem_get_used_fast`](/utils.mem.html#gpu_mem_get_used_fast)

show_doc(gpu_with_max_free_mem)


# [`gpu_with_max_free_mem`](/utils.mem.html#gpu_with_max_free_mem):
# * for gpu returns: `gpu_with_max_free_ram_id, its_free_ram`
# * for cpu returns: `None, 0`
#

show_doc(preload_pytorch)


# [`preload_pytorch`](/utils.mem.html#preload_pytorch) is helpful when GPU memory is being measured, since the first time any operation on `cuda` is performed by pytorch, usually about 0.5GB gets used by CUDA context.

show_doc(GPUMemory, title_level=4)


# [`GPUMemory`](/utils.mem.html#GPUMemory) is a namedtuple that is returned by functions like [`gpu_mem_get`](/utils.mem.html#gpu_mem_get) and [`gpu_mem_get_all`](/utils.mem.html#gpu_mem_get_all).

show_doc(b2mb)


# [`b2mb`](/utils.mem.html#b2mb) is a helper utility that just does `int(bytes/2**20)`

# ## Memory Tracing Utils

show_doc(GPUMemTrace, title_level=4)


# **Arguments**:
#
# * `silent`: a shortcut to make `report` and `report_n_reset` silent w/o needing to remove those calls - this can be done from the constructor, or alternatively you can call `silent` method anywhere to do the same.
# * `ctx`: default context note in reports
# * `on_exit_report`:  auto-report on ctx manager exit (default `True`)
#
# **Definitions**:
#
# * **Delta Used** is the difference between current used memory and used memory at the start of the counter.
#
# * **Delta Peaked** is the memory overhead if any. It's calculated in two steps:
#    1. The base measurement is the difference between the peak memory and the used memory at the start of the counter.
#    2. Then if delta used is positive it gets subtracted from the base value.
#
#    It indicates the size of the blip.
#
#    **Warning**: currently the peak memory usage tracking is implemented using a python thread, which is very unreliable, since there is no guarantee the thread will get a chance at running at the moment the peak memory is occuring (or it might not get a chance to run at all). Therefore we need pytorch to implement multiple concurrent and resettable [`torch.cuda.max_memory_allocated`](https://pytorch.org/docs/stable/cuda.html#torch.cuda.max_memory_allocated) counters. Please vote for this [feature request](https://github.com/pytorch/pytorch/issues/16266).
#
# **Usage Examples**:
#
# Setup:
# ```
# from fastai.utils.mem import GPUMemTrace
# def some_code(): pass
# mtrace = GPUMemTrace()
# ```
#
# Example 1: basic measurements via `report` (prints) and via [`data`](/tabular.data.html#tabular.data) (returns) accessors
# ```
# some_code()
# mtrace.report()
# delta_used, delta_peaked = mtrace.data()
#
# some_code()
# mtrace.report('2nd run of some_code()')
# delta_used, delta_peaked = mtrace.data()
# ```
# `report`'s optional `subctx` argument can be helpful if you have many `report` calls and you want to understand which is which in the outputs.
#
# Example 2: measure in a loop, resetting the counter before each run
# ```
# for i in range(10):
#     mtrace.reset()
#     some_code()
#     mtrace.report(f'i={i}')
# ```
# `reset` resets all the counters.
#
# Example 3: like example 2, but having `report` automatically reset the counters
# ```
# mtrace.reset()
# for i in range(10):
#     some_code()
#     mtrace.report_n_reset(f'i={i}')
# ```
#
# The tracing starts immediately upon the [`GPUMemTrace`](/utils.mem.html#GPUMemTrace) object creation, and stops when that object is deleted. But it can also be `stop`ed, `start`ed manually as well.
# ```
# mtrace.start()
# mtrace.stop()
# ```
# `stop` is in particular useful if you want to **freeze** the [`GPUMemTrace`](/utils.mem.html#GPUMemTrace) object and to be able to query its data on `stop` some time down the road.
#
#
# **Reporting**:
#
# In reports you can print a main context passed via the constructor:
#
# ```
# mtrace = GPUMemTrace(ctx="foobar")
# mtrace.report()
# ```
# prints:
# ```
# △Used Peaked MB:      0      0  (foobar)
# ```
#
# and then add subcontext notes as needed:
#
# ```
# mtrace = GPUMemTrace(ctx="foobar")
# mtrace.report('1st try')
# mtrace.report('2nd try')
#
# ```
# prints:
# ```
# △Used Peaked MB:      0      0  (foobar: 1st try)
# △Used Peaked MB:      0      0  (foobar: 2nd try)
# ```
#
# Both context and sub-context are optional, and are very useful if you sprinkle [`GPUMemTrace`](/utils.mem.html#GPUMemTrace) in different places around the code.
#
# You can silence report calls w/o needing to remove them via constructor or `silent`:
#
# ```
# mtrace = GPUMemTrace(silent=True)
# mtrace.report() # nothing will be printed
# mtrace.silent(silent=False)
# mtrace.report() # printing resumed
# mtrace.silent(silent=True)
# mtrace.report() # nothing will be printed
# ```
#
# **Context Manager**:
#
# [`GPUMemTrace`](/utils.mem.html#GPUMemTrace) can also be used as a context manager:
#
# Report the used and peaked deltas automatically:
#
# ```
# with GPUMemTrace(): some_code()
# ```
#
# If you wish to add context:
#
# ```
# with GPUMemTrace(ctx='some context'): some_code()
# ```
#
# The context manager uses subcontext `exit` to indicate that the report comes after the context exited.
#
# The reporting is done automatically, which is especially useful in functions due to return call:
#
# ```
# def some_func():
#     with GPUMemTrace(ctx='some_func'):
#         # some code
#         return 1
# some_func()
# ```
# prints:
# ```
# △Used Peaked MB:      0      0 (some_func: exit)
# ```
# so you still get a perfect report despite the `return` call here. `ctx` is useful for specifying the *context* in case you have many of those calls through your code and you want to know which is which.
#
# And, of course, instead of doing the above, you can use [`gpu_mem_trace`](/utils.mem.html#gpu_mem_trace) decorator to do it automatically, including using the function or method name as the context. Therefore, the example below does the same without modifying the function.
#
# ```
# @gpu_mem_trace
# def some_func():
#     # some code
#     return 1
# some_func()
# ```
#
# If you don't wish the automatic reporting, just pass `on_exit_report=False` in the constructor:
#
# ```
# with GPUMemTrace(ctx='some_func', on_exit_report=False) as mtrace:
#     some_code()
# mtrace.report("measured in ctx")
# ```
#
# or the same w/o the context note:
# ```
# with GPUMemTrace(on_exit_report=False) as mtrace: some_code()
# print(mtrace) # or mtrace.report()
# ```
#
# And, of course, you can get the numerical data (in rounded MBs):
# ```
# with GPUMemTrace() as mtrace: some_code()
# delta_used, delta_peaked = mtrace.data()
# ```

show_doc(gpu_mem_trace)


# This allows you to decorate any function or method with:
#
# ```
# @gpu_mem_trace
# def my_function(): pass
# # run:
# my_function()
# ```
# and it will automatically print the report including the function name as a context:
# ```
# △Used Peaked MB:      0      0 (my_function: exit)
# ```
# In the case of methods it'll print a fully qualified method, e.g.:
# ```
# △Used Peaked MB:      0      0 (Class.function: exit)
# ```
#

# ## Undocumented Methods - Methods moved below this line will intentionally be hidden

show_doc(GPUMemTrace.report)


show_doc(GPUMemTrace.silent)


show_doc(GPUMemTrace.start)


show_doc(GPUMemTrace.reset)


show_doc(GPUMemTrace.peak_monitor_stop)


show_doc(GPUMemTrace.stop)


show_doc(GPUMemTrace.report_n_reset)


show_doc(GPUMemTrace.peak_monitor_func)


show_doc(GPUMemTrace.data_set)


show_doc(GPUMemTrace.data)


show_doc(GPUMemTrace.peak_monitor_start)


# ## New Methods - Please document or move to the undocumented section
