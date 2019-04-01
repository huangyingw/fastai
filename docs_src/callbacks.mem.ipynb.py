
# coding: utf-8

# ## callbacks.mem

# Memory profiling callbacks.

from fastai.gen_doc.nbdoc import *
from fastai.callbacks.mem import *


show_doc(PeakMemMetric)


# [`PeakMemMetric`](/callbacks.mem.html#PeakMemMetric) is a memory profiling callback.
#
# Here is how you can use it:
#
# ```
# from fastai.callbacks.mem import PeakMemMetric
# learn = cnn_learner(data, model, metrics=[accuracy], callback_fns=PeakMemMetric)
# learn.fit_one_cycle(3, max_lr=1e-2)
# ```
#
# and a sample output:
# ```
# Total time: 00:59
# epoch	train_loss valid_loss accuracy cpu used peak gpu used peak
#     1	0.325806   0.070334   0.978800	      0   2       80  6220
#     2	0.093147   0.038905   0.987700	      0   2        2   914
#     3	0.047818   0.027617   0.990600	      0   2        0   912
# ```
#
# The last four columns are deltas memory usage for CPU and GPU (in MBs).
#
# * The "used memory" columns show the difference between memory usage before and after each epoch.
# * The "peaked memory" columns how much memory overhead the epoch used on top of used memory. With the rare exception of gpu measurements, where if "used memory" delta is negative, then it's calculated as a straight difference between the peak memory and the used memory at the beginning of the epoch. Also see
#
# For example in the first row of the above sample example it shows `used=80`, `peak=6220`. It means that during the execution of this thread the application used a maximum of 6300 MBs (`80+6220`), but then most of that memory was released, keeping only 80 MBs tied up. You can then see in the following epochs that while the application still uses temporary memory while execution, but it releases almost all of it at the end of its work.
#
# Also, it's very important to know that pytorch's memory allocator can work with less memory, so it doesn't mean that it needs 6300 MB to be able to run the first epoch. It will do with less, but it will just be slightly slower on the first epoch. For more details please see [this explanation](dev/gpu.html#peak-memory-usage).
#

# ## Undocumented Methods - Methods moved below this line will intentionally be hidden
