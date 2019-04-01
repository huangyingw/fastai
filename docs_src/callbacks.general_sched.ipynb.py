
# coding: utf-8

# ## TrainingPhase and General scheduler

# Creates a scheduler that lets you train a model with following different [`TrainingPhase`](/callbacks.general_sched.html#TrainingPhase).

from fastai.gen_doc.nbdoc import *
from fastai.callbacks.general_sched import *
from fastai.vision import *


show_doc(TrainingPhase)


# You can then schedule any hyper-parameter you want by using the following method.

show_doc(TrainingPhase.schedule_hp)


# The phase will make the hyper-parameter vary from the first value in `vals` to the second, following `anneal`. If an annealing function is specified but `vals` is a float, it will decay to 0. If no annealing function is specified, the default is a linear annealing for a tuple, a constant parameter if it's a float.

jekyll_note("""If you want to use discriminative values, you can pass an numpy array in `vals` (or a tuple
of them for start and stop).""")


# The basic hyper-parameters are named:
# - 'lr' for learning rate
# - 'mom' for momentum (or beta1 in Adam)
# - 'beta' for the beta2 in Adam or the alpha in RMSprop
# - 'wd' for weight decay
#
# You can also add any hyper-parameter that is in your optimizer (even if it's custom or a [`GeneralOptimizer`](/general_optimizer.html#GeneralOptimizer)), like 'eps' if you're using Adam.

# Let's make an example by using this to code [SGD with warm restarts](https://arxiv.org/abs/1608.03983).

def fit_sgd_warm(learn, n_cycles, lr, mom, cycle_len, cycle_mult):
    n = len(learn.data.train_dl)
    phases = [(TrainingPhase(n * (cycle_len * cycle_mult**i))
                 .schedule_hp('lr', lr, anneal=annealing_cos)
                 .schedule_hp('mom', mom)) for i in range(n_cycles)]
    sched = GeneralScheduler(learn, phases)
    learn.callbacks.append(sched)
    if cycle_mult != 1:
        total_epochs = int(cycle_len * (1 - (cycle_mult)**n_cycles) / (1 - cycle_mult))
    else: total_epochs = n_cycles * cycle_len
    learn.fit(total_epochs)


path = untar_data(URLs.MNIST_SAMPLE)
data = ImageDataBunch.from_folder(path)
learn = Learner(data, simple_cnn((3, 16, 16, 2)), metrics=accuracy)
fit_sgd_warm(learn, 3, 1e-3, 0.9, 1, 2)


learn.recorder.plot_lr()


show_doc(GeneralScheduler)


# ### Callback methods

# You don't call these yourself - they're called by fastai's [`Callback`](/callback.html#Callback) system automatically to enable the class's functionality.

show_doc(GeneralScheduler.on_batch_end, doc_string=False)


# Takes a step in the current phase and prepare the hyperparameters for the next batch.

show_doc(GeneralScheduler.on_train_begin, doc_string=False)


# Initiates the hyperparameters to the start values of the first phase.

# ## Undocumented Methods - Methods moved below this line will intentionally be hidden
