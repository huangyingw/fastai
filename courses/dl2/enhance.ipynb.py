
# coding: utf-8

# **Important: This notebook will only work with fastai-0.7.x. Do not try to run any fastai-1.x code from this path in the repository because it will load fastai-0.7.x**

get_ipython().run_line_magic('matplotlib', 'inline')
get_ipython().run_line_magic('reload_ext', 'autoreload')
get_ipython().run_line_magic('autoreload', '2')


get_ipython().run_line_magic('pinfo', 're.compile')


# ## Super resolution data

from fastai.conv_learner import *
from pathlib import Path
torch.cuda.set_device(0)

torch.backends.cudnn.benchmark = True


PATH = Path('data/imagenet')
PATH_TRN = PATH / 'train'


fnames_full, label_arr_full, all_labels = folder_source(PATH, 'train')
fnames_full = ['/'.join(Path(fn).parts[-2:]) for fn in fnames_full]
list(zip(fnames_full[:5], label_arr_full[:5]))


all_labels[:5]


np.random.seed(42)
# keep_pct = 1.
keep_pct = 0.02
keeps = np.random.rand(len(fnames_full)) < keep_pct
fnames = np.array(fnames_full, copy=False)[keeps]
label_arr = np.array(label_arr_full, copy=False)[keeps]


arch = vgg16
sz_lr = 72


# scale,bs = 2,64
scale, bs = 4, 24
sz_hr = sz_lr * scale


class MatchedFilesDataset(FilesDataset):
    def __init__(self, fnames, y, transform, path):
        self.y = y
        assert(len(fnames) == len(y))
        super().__init__(fnames, transform, path)

    def get_y(self, i): return open_image(os.path.join(self.path, self.y[i]))

    def get_c(self): return 0


aug_tfms = [RandomDihedral(tfm_y=TfmType.PIXEL)]


val_idxs = get_cv_idxs(len(fnames), val_pct=min(0.01 / keep_pct, 0.1))
((val_x, trn_x), (val_y, trn_y)) = split_by_idx(val_idxs, np.array(fnames), np.array(fnames))
len(val_x), len(trn_x)


img_fn = PATH / 'train' / 'n01558993' / 'n01558993_9684.JPEG'


tfms = tfms_from_model(arch, sz_lr, tfm_y=TfmType.PIXEL, aug_tfms=aug_tfms, sz_y=sz_hr)
datasets = ImageData.get_ds(MatchedFilesDataset, (trn_x, trn_y), (val_x, val_y), tfms, path=PATH_TRN)
md = ImageData(PATH, datasets, bs, num_workers=16, classes=None)


denorm = md.val_ds.denorm


def show_img(ims, idx, figsize=(5, 5), normed=True, ax=None):
    if ax is None: fig, ax = plt.subplots(figsize=figsize)
    if normed: ims = denorm(ims)
    else: ims = np.rollaxis(to_np(ims), 1, 4)
    ax.imshow(np.clip(ims, 0, 1)[idx])
    ax.axis('off')


x, y = next(iter(md.val_dl))
x.size(), y.size()


idx = 1
fig, axes = plt.subplots(1, 2, figsize=(9, 5))
show_img(x, idx, ax=axes[0])
show_img(y, idx, ax=axes[1])


batches = [next(iter(md.aug_dl)) for i in range(9)]


fig, axes = plt.subplots(3, 6, figsize=(18, 9))
for i, (x, y) in enumerate(batches):
    show_img(x, idx, ax=axes.flat[i * 2])
    show_img(y, idx, ax=axes.flat[i * 2 + 1])


# ## Model

def conv(ni, nf, kernel_size=3, actn=True):
    layers = [nn.Conv2d(ni, nf, kernel_size, padding=kernel_size // 2)]
    if actn: layers.append(nn.ReLU(True))
    return nn.Sequential(*layers)


class ResSequential(nn.Module):
    def __init__(self, layers, res_scale=1.0):
        super().__init__()
        self.res_scale = res_scale
        self.m = nn.Sequential(*layers)

    def forward(self, x):
        x = x + self.m(x) * self.res_scale
        return x


def res_block(nf):
    return ResSequential(
        [conv(nf, nf), conv(nf, nf, actn=False)],
        0.1)


def upsample(ni, nf, scale):
    layers = []
    for i in range(int(math.log(scale, 2))):
        layers += [conv(ni, nf * 4), nn.PixelShuffle(2)]
    return nn.Sequential(*layers)


class SrResnet(nn.Module):
    def __init__(self, nf, scale):
        super().__init__()
        features = [conv(3, 64)]
        for i in range(8): features.append(res_block(64))
        features += [conv(64, 64), upsample(64, 64, scale),
                     nn.BatchNorm2d(64),
                     conv(64, 3, actn=False)]
        self.features = nn.Sequential(*features)
        
    def forward(self, x): return self.features(x)


# ## Pixel loss

m = to_gpu(SrResnet(64, scale))
m = nn.DataParallel(m, [0, 2])
learn = Learner(md, SingleModel(m), opt_fn=optim.Adam)
learn.crit = F.mse_loss


learn.lr_find(start_lr=1e-5, end_lr=10000)
learn.sched.plot()


lr = 2e-3


learn.fit(lr, 1, cycle_len=1, use_clr_beta=(40, 10))


x, y = next(iter(md.val_dl))
preds = learn.model(VV(x))


idx = 4
show_img(y, idx, normed=False)


show_img(preds, idx, normed=False);


show_img(x, idx, normed=True);


x, y = next(iter(md.val_dl))
preds = learn.model(VV(x))


show_img(y, idx, normed=False)


show_img(preds, idx, normed=False);


show_img(x, idx);


# ## Perceptual loss

def icnr(x, scale=2, init=nn.init.kaiming_normal):
    new_shape = [int(x.shape[0] / (scale ** 2))] + list(x.shape[1:])
    subkernel = torch.zeros(new_shape)
    subkernel = init(subkernel)
    subkernel = subkernel.transpose(0, 1)
    subkernel = subkernel.contiguous().view(subkernel.shape[0],
                                            subkernel.shape[1], -1)
    kernel = subkernel.repeat(1, 1, scale ** 2)
    transposed_shape = [x.shape[1]] + [x.shape[0]] + list(x.shape[2:])
    kernel = kernel.contiguous().view(transposed_shape)
    kernel = kernel.transpose(0, 1)
    return kernel


m_vgg = vgg16(True)

blocks = [i - 1 for i, o in enumerate(children(m_vgg))
              if isinstance(o, nn.MaxPool2d)]
blocks, [m_vgg[i] for i in blocks]


vgg_layers = children(m_vgg)[:13]
m_vgg = nn.Sequential(*vgg_layers).cuda().eval()
set_trainable(m_vgg, False)


def flatten(x): return x.view(x.size(0), -1)


class SaveFeatures():
    features = None

    def __init__(self, m): self.hook = m.register_forward_hook(self.hook_fn)

    def hook_fn(self, module, input, output): self.features = output

    def remove(self): self.hook.remove()


class FeatureLoss(nn.Module):
    def __init__(self, m, layer_ids, layer_wgts):
        super().__init__()
        self.m, self.wgts = m, layer_wgts
        self.sfs = [SaveFeatures(m[i]) for i in layer_ids]

    def forward(self, input, target, sum_layers=True):
        self.m(VV(target.data))
        res = [F.l1_loss(input, target) / 100]
        targ_feat = [V(o.features.data.clone()) for o in self.sfs]
        self.m(input)
        res += [F.l1_loss(flatten(inp.features), flatten(targ)) * wgt
               for inp, targ, wgt in zip(self.sfs, targ_feat, self.wgts)]
        if sum_layers: res = sum(res)
        return res
    
    def close(self):
        for o in self.sfs: o.remove()


m = SrResnet(64, scale)


conv_shuffle = m.features[10][0][0]
kernel = icnr(conv_shuffle.weight, scale=scale)
conv_shuffle.weight.data.copy_(kernel);


conv_shuffle = m.features[10][2][0]
kernel = icnr(conv_shuffle.weight, scale=scale)
conv_shuffle.weight.data.copy_(kernel);


m = to_gpu(m)


learn = Learner(md, SingleModel(m), opt_fn=optim.Adam)


t = torch.load(learn.get_model_path('sr-samp0'), map_location=lambda storage, loc: storage)
learn.model.load_state_dict(t, strict=False)


learn.freeze_to(999)


for i in range(10, 13): set_trainable(m.features[i], True)


lr = 6e-3
wd = 1e-7


learn.fit(lr, 1, cycle_len=1, wds=wd, use_clr=(20, 10))


learn.crit = FeatureLoss(m_vgg, blocks[:2], [0.26, 0.74])


learn.lr_find(1e-4, 1., wds=wd, linear=True)


learn.load('tmp')


learn.sched.plot(0, n_skip_end=1)


learn.save('sr-samp0')


learn.unfreeze()


learn.fit(lr, 1, cycle_len=1, wds=wd, use_clr=(20, 10))


learn.fit(lr, 1, cycle_len=2, wds=wd, use_clr=(20, 10))


learn.fit(lr, 1, cycle_len=2, wds=wd, use_clr=(20, 10))


learn.sched.plot_loss()


learn.load('sr-samp1')


learn.save('sr-samp1')


learn.load('sr-samp1')


lr = 3e-3


learn.fit(lr, 1, cycle_len=1, wds=wd, use_clr=(20, 10))


learn.save('sr-samp2')


learn.unfreeze()


learn.load('sr-samp2')


learn.fit(lr / 3, 1, cycle_len=1, wds=wd, use_clr=(20, 10))


learn.save('sr1')


def plot_ds_img(idx, ax=None, figsize=(7, 7), normed=True):
    if ax is None: fig, ax = plt.subplots(figsize=figsize)
    im = md.val_ds[idx][0]
    if normed: im = denorm(im)[0]
    else: im = np.rollaxis(to_np(im), 0, 3)
    ax.imshow(im)
    ax.axis('off')


fig, axes = plt.subplots(6, 6, figsize=(20, 20))
for i, ax in enumerate(axes.flat): plot_ds_img(i + 200, ax=ax, normed=True)


x, y = md.val_ds[201]


y = y[None]


learn.model.eval()
preds = learn.model(VV(x[None]))
x.shape, y.shape, preds.shape


learn.crit(preds, V(y), sum_layers=False)


learn.crit(preds, V(y), sum_layers=False)


learn.crit.close()


_, axes = plt.subplots(1, 2, figsize=(14, 7))
show_img(x[None], 0, ax=axes[0])
show_img(preds, 0, normed=True, ax=axes[1])


_, axes = plt.subplots(1, 2, figsize=(14, 7))
show_img(x[None], 0, ax=axes[0])
show_img(preds, 0, normed=True, ax=axes[1])


_, axes = plt.subplots(1, 2, figsize=(14, 7))
show_img(x[None], 0, ax=axes[0])
show_img(preds, 0, normed=True, ax=axes[1])
