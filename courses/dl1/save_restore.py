from fastai.conv_learner import *
from fastai.dataset import *
from fastai.imports import *
from fastai.model import *
from fastai.plots import *
from fastai.sgdr import *
from fastai.transforms import *
import os
import os.path
import subprocess
import torch

os.chdir(os.path.dirname(os.path.realpath(__file__)))
PATH = "data/smallset/"
sz = 10
torch.cuda.is_available()
torch.backends.cudnn.enabled

command = "ls %svalid/cats | head" % (PATH)
files = subprocess.getoutput(command).split()

file_name = "%svalid/cats/%s" % (PATH, files[0])
img = plt.imread(file_name)
# plt.imshow(img)

# Here is how the raw data looks like
img.shape
img[:4, :4]

# Uncomment the below if you need to reset your precomputed activations
command = "rm -rf %stmp" % (PATH)
subprocess.getoutput(command)

arch = resnet34
data = ImageClassifierData.from_paths(PATH, tfms=tfms_from_model(arch, sz))


tfms = tfms_from_model(
    resnet34, sz, aug_tfms=transforms_side_on, max_zoom=1.1)

def learn1():
    data = ImageClassifierData.from_paths(PATH, tfms=tfms)
    learn = ConvLearner.pretrained(arch, data, precompute=True)
    learn.lr_find()
    learn.sched.plot_lr()
    learn.sched.plot()
    #learn.fit(1e-2, 1, saved_model_name='save_restore')
    #learn.sched.plot_lr()

learn1()
