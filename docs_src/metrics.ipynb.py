
# coding: utf-8

# ## Training metrics

# *Metrics* for training fastai models are simply functions that take `input` and `target` tensors, and return some metric of interest for training. You can write your own metrics by defining a function of that type, and passing it to [`Learner`](/basic_train.html#Learner) in the [`metrics`](/metrics.html#metrics) parameter, or use one of the following pre-defined functions.

from fastai.gen_doc.nbdoc import *
from fastai.basics import *


# ## Predefined metrics:

show_doc(accuracy)


jekyll_warn("This metric is intended for classification of objects belonging to a single class.")


show_doc(accuracy_thresh)


# Prediction are compared to `thresh` after `sigmoid` is maybe applied. Then we count the numbers that match the targets.

jekyll_note("This function is intended for one-hot-encoded targets (often in a multiclassification problem).")


show_doc(top_k_accuracy)


show_doc(dice)


show_doc(error_rate)


show_doc(mean_squared_error)


show_doc(mean_absolute_error)


show_doc(mean_squared_logarithmic_error)


show_doc(exp_rmspe)


show_doc(root_mean_squared_error)


show_doc(fbeta)


# `beta` determines the value of the fbeta applied, `eps` is there for numeric stability. If `sigmoid=True`, a sigmoid is applied to the predictions before comparing them to `thresh` then to the targets. See the [F1 score wikipedia page](https://en.wikipedia.org/wiki/F1_score) for details on the fbeta score.

jekyll_note("This function is intended for one-hot-encoded targets (often in a multiclassification problem).")


show_doc(explained_variance)


show_doc(r2_score)


# The following metrics are classes, don't forget to instantiate them when you pass them to a [`Learner`](/basic_train.html#Learner).

show_doc(RMSE, title_level=3)


show_doc(ExpRMSPE, title_level=3)


show_doc(Precision, title_level=3)


show_doc(Recall, title_level=3)


show_doc(FBeta, title_level=3)


show_doc(R2Score, title_level=3)


show_doc(ExplainedVariance, title_level=3)


show_doc(MatthewsCorreff, title_level=3)


# Ref.: https://github.com/scikit-learn/scikit-learn/blob/bac89c2/sklearn/metrics/classification.py

show_doc(KappaScore, title_level=3)


# Ref.: https://github.com/scikit-learn/scikit-learn/blob/bac89c2/sklearn/metrics/classification.py
#
# [`KappaScore`](/metrics.html#KappaScore) supports linear and quadratic weights on the off-diagonal cells in the [`ConfusionMatrix`](/metrics.html#ConfusionMatrix), in addition to the default unweighted calculation treating all misclassifications as equally weighted. Leaving [`KappaScore`](/metrics.html#KappaScore)'s `weights` attribute as `None` returns the unweighted Kappa score. Updating `weights` to "linear" means off-diagonal ConfusionMatrix elements are weighted in linear proportion to their distance from the diagonal; "quadratic" means weights are squared proportional to their distance from the diagonal.
# Specify linear or quadratic weights, if using, by first creating an instance of the metric and then updating the `weights` attribute, similar to as follows:
# ```
# kappa = KappaScore()
# kappa.weights = "quadratic"
# learn = cnn_learner(data, model, metrics=[error_rate, kappa])
# ```

show_doc(ConfusionMatrix, title_level=3)


# ## Creating your own metric

# Creating a new metric can be as simple as creating a new function. If your metric is an average over the total number of elements in your dataset, just write the function that will compute it on a batch (taking `pred` and `targ` as arguments). It will then be automatically averaged over the batches (taking their different sizes into account).
#
# Sometimes metrics aren't simple averages however. If we take the example of precision for instance, we have to divide the number of true positives by the number of predictions we made for that class. This isn't an average over the number of elements we have in the dataset, we only consider those where we made a positive prediction for a specific thing. Computing the precision for each batch, then averaging them will yield to a result that may be close to the real value, but won't be it exactly (and it really depends on how you deal with special case of 0 positive predictions).
#
# This why in fastai, every metric is implemented as a callback. If you pass a regular function, the library transforms it to a proper callback called `AverageCallback`. The callback metrics are only called during the validation phase, and only for the following events:
# - <code>on_epoch_begin</code> (for initialization)
# - <code>on_batch_begin</code> (if we need to have a look at the input/target and maybe modify them)
# - <code>on_batch_end</code> (to analyze the last results and update our computation)
# - <code>on_epoch_end</code>(to wrap up the final result that should be added to `last_metrics`)
#
# As an example, the following code is the exact implementation of the [`AverageMetric`](/callback.html#AverageMetric) callback that transforms a function like [`accuracy`](/metrics.html#accuracy) into a metric callback.

class AverageMetric(Callback):
    "Wrap a `func` in a callback for metrics computation."

    def __init__(self, func):
        # If it's a partial, use func.func
        name = getattr(func, 'func', func).__name__
        self.func, self.name = func, name

    def on_epoch_begin(self, **kwargs):
        "Set the inner value to 0."
        self.val, self.count = 0., 0

    def on_batch_end(self, last_output, last_target, **kwargs):
        "Update metric computation with `last_output` and `last_target`."
        if not is_listy(last_target): last_target = [last_target]
        self.count += last_target[0].size(0)
        val = self.func(last_output, *last_target)
        self.val += last_target[0].size(0) * val.detach().cpu()

    def on_epoch_end(self, last_metrics, **kwargs):
        "Set the final result in `last_metrics`."
        return add_metrics(last_metrics, self.val / self.count)


# Here [`add_metrics`](/torch_core.html#add_metrics) is a convenience function that will return the proper dictionary for us:
# ```python
# {'last_metrics': last_metrics + [self.val/self.count]}
# ```

# And here is another example that properly computes the precision for a given class.

class Precision(Callback):
    
    def on_epoch_begin(self, **kwargs):
        self.correct, self.total = 0, 0
    
    def on_batch_end(self, last_output, last_target, **kwargs):
        preds = last_output.argmax(1)
        self.correct += ((preds == 0) * (last_target == 0)).float().sum()
        self.total += (preds == 0).float().sum()
    
    def on_epoch_end(self, last_metrics, **kwargs):
        return add_metrics(last_metrics, self.correct / self.total)


# The following custom callback class example measures peak RAM usage during each epoch:

import tracemalloc
class TraceMallocMetric(Callback):
    def __init__(self):
        super().__init__()
        self.name = "peak RAM"

    def on_epoch_begin(self, **kwargs):
        tracemalloc.start()
        
    def on_epoch_end(self, last_metrics, **kwargs):
        current, peak = tracemalloc.get_traced_memory()
        tracemalloc.stop()
        return add_metrics(last_metrics, torch.tensor(peak))


# To deploy it, you need to pass an instance of this custom metric in the [`metrics`](/metrics.html#metrics) argument:
#
# ```
# learn = cnn_learner(data, model, metrics=[accuracy, TraceMallocMetric()])
# learn.fit_one_cycle(3, max_lr=1e-2)
# ```
# And then the output changes to:
# ```
# Total time: 00:54
# epoch	train_loss	valid_loss	accuracy	peak RAM
#    1	0.333352	0.084342	0.973800	2395541.000000
#    2	0.096196	0.038386	0.988300	2342145.000000
#    3	0.048722	0.029234	0.990200	2342680.000000
# ```
#

# As mentioner earlier, using the [`metrics`](/metrics.html#metrics) argument with a custom metrics class is limited in the number of phases of the callback system it can access, it can only return one numerical value and as you can see its output is hardcoded to have 6 points of precision in the output, even if the number is an int.
#
# To overcome these limitations callback classes should be used instead.
#
# For example, the following class:
# * uses phases not available for the metric classes
# * it reports 3 columns, instead of just one
# * its column report ints, instead of floats

import tracemalloc
class TraceMallocMultiColMetric(LearnerCallback):
    _order = -20 # Needs to run before the recorder

    def __init__(self, learn):
        super().__init__(learn)
        self.train_max = 0

    def on_train_begin(self, **kwargs):
        self.learn.recorder.add_metric_names(['used', 'max_used', 'peak'])
            
    def on_batch_end(self, train, **kwargs):
        # track max memory usage during the train phase
        if train:
            current, peak = tracemalloc.get_traced_memory()
            self.train_max = max(self.train_max, current)
        
    def on_epoch_begin(self, **kwargs):
        tracemalloc.start()

    def on_epoch_end(self, last_metrics, **kwargs):
        current, peak = tracemalloc.get_traced_memory()
        tracemalloc.stop()
        return add_metrics(last_metrics, [current, self.train_max, peak])


# Note, that it subclasses [`LearnerCallback`](/basic_train.html#LearnerCallback) and not [`Callback`](/callback.html#Callback), since the former provides extra features not available in the latter.
#
# Also `_order=-20` is crucial - without it the custom columns will not be added - it tells the callback system to run this callback before the recorder system.
#
# To deploy it, you need to pass the name of the class (not an instance!) of the class in the `callback_fns` argument. This is because the `learn` object doesn't exist yet, and it's required to instantiate `TraceMallocMultiColMetric`. The system will do it for us automatically as soon as the learn object has been created.
#
# ```
# learn = cnn_learner(data, model, metrics=[accuracy], callback_fns=TraceMallocMultiColMetric)
# learn.fit_one_cycle(3, max_lr=1e-2)
# ```
# And then the output changes to:
# ```
# Total time: 00:53
# epoch	train_loss valid_loss   accuracy	 used	max_used   peak
#     1	0.321233	0.068252	0.978600	156504	2408404	  2419891
#     2	0.093551	0.032776	0.988500	 79343	2408404	  2348085
#     3	0.047178	0.025307	0.992100	 79568	2408404	  2342754
# ```
#
# Another way to do the same is by using `learn.callbacks.append`, and this time we need to instantiate `TraceMallocMultiColMetric` with `learn` object which we now have, as it is called after the latter was created:
#
# ```
# learn = cnn_learner(data, model, metrics=[accuracy])
# learn.callbacks.append(TraceMallocMultiColMetric(learn))
# learn.fit_one_cycle(3, max_lr=1e-2)
# ```
#
# Configuring the custom metrics in the `learn` object sets them to run in all future [`fit`](/basic_train.html#fit)-family calls. However, if you'd like to configure it for just one call, you can configure it directly inside [`fit`](/basic_train.html#fit) or [`fit_one_cycle`](/train.html#fit_one_cycle):
#
# ```
# learn = cnn_learner(data, model, metrics=[accuracy])
# learn.fit_one_cycle(3, max_lr=1e-2, callbacks=TraceMallocMultiColMetric(learn))
# ```
#
# And to stress the differences:
# * the `callback_fns` argument expects a classname or a list of those
# * the [`callbacks`](/callbacks.html#callbacks) argument expects an instance of a class or a list of those
# * `learn.callbacks.append` expects a single instance of a class
#
# For more examples, look inside fastai codebase and its test suite, search for classes that subclass either [`Callback`](/callback.html#Callback), [`LearnerCallback`](/basic_train.html#LearnerCallback) and subclasses of those two.
#
# Finally, while the above examples all add to the metrics, it's not a requirement. A callback can do anything it wants and it is not required to add its outcomes to the metrics printout.

# ## Undocumented Methods - Methods moved below this line will intentionally be hidden

show_doc(FBeta.on_batch_end)


show_doc(FBeta.on_epoch_begin)


show_doc(FBeta.on_epoch_end)


show_doc(mae)


show_doc(msle)


show_doc(mse)


show_doc(rmse)


show_doc(Precision.on_epoch_end)


show_doc(FBeta.on_train_end)


show_doc(KappaScore.on_epoch_end)


show_doc(MatthewsCorreff.on_epoch_end)


show_doc(FBeta.on_train_begin)


show_doc(RMSE.on_epoch_end)


show_doc(ConfusionMatrix.on_train_begin)


show_doc(ConfusionMatrix.on_batch_end)


show_doc(ConfusionMatrix.on_epoch_end)


show_doc(Recall.on_epoch_end)


show_doc(ExplainedVariance.on_epoch_end)


show_doc(ExpRMSPE.on_epoch_end)


show_doc(ConfusionMatrix.on_epoch_begin)


show_doc(R2Score.on_epoch_end)


# ## New Methods - Please document or move to the undocumented section
