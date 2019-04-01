
# coding: utf-8

# ## The data block API

from fastai.gen_doc.nbdoc import *
from fastai.basics import *
np.random.seed(42)


# The data block API lets you customize the creation of a [`DataBunch`](/basic_data.html#DataBunch) by isolating the underlying parts of that process in separate blocks, mainly:
#   1. Where are the inputs and how to create them?
#   1. How to split the data into a training and validation sets?
#   1. How to label the inputs?
#   1. What transforms to apply?
#   1. How to add a test set?
#   1. How to wrap in dataloaders and create the [`DataBunch`](/basic_data.html#DataBunch)?
#
# Each of these may be addressed with a specific block designed for your unique setup. Your inputs might be in a folder, a csv file, or a dataframe. You may want to split them randomly, by certain indices or depending on the folder they are in. You can have your labels in your csv file or your dataframe, but it may come from folders or a specific function of the input. You may choose to add data augmentation or not. A test set is optional too. Finally you have to set the arguments to put the data together in a [`DataBunch`](/basic_data.html#DataBunch) (batch size, collate function...)
#
# The data block API is called as such because you can mix and match each one of those blocks with the others, allowing for a total flexibility to create your customized [`DataBunch`](/basic_data.html#DataBunch) for training, validation and testing. The factory methods of the various [`DataBunch`](/basic_data.html#DataBunch) are great for beginners but you can't always make your data fit in the tracks they require.
#
# <img src="imgs/mix_match.png" alt="Mix and match" width="200">
#
# As usual, we'll begin with end-to-end examples, then switch to the details of each of those parts.

# ## Examples of use

# Let's begin with our traditional MNIST example.

from fastai.vision import *


path = untar_data(URLs.MNIST_TINY)
tfms = get_transforms(do_flip=False)
path.ls()


(path / 'train').ls()


# In [`vision.data`](/vision.data.html#vision.data), we can create a [`DataBunch`](/basic_data.html#DataBunch) suitable for image classification by simply typing:

data = ImageDataBunch.from_folder(path, ds_tfms=tfms, size=64)


# This is a shortcut method which is aimed at data that is in folders following an ImageNet style, with the [`train`](/train.html#train) and `valid` directories, each containing one subdirectory per class, where all the labelled pictures are. There is also a `test` directory containing unlabelled pictures.
#
# Here is the same code, but this time using the data block API, which can work with any style of a dataset. All the stages, which will be explained below, can be grouped together like this:

data = (ImageList.from_folder(path) #Where to find the data? -> in path and its subfolders
        .split_by_folder()              #How to split in train/valid? -> use the folders
        .label_from_folder()            #How to label? -> depending on the folder of the filenames
        .add_test_folder()              #Optionally add a test set (here default name is test)
        .transform(tfms, size=64)       #Data augmentation? -> use tfms with a size of 64
        .databunch())                   #Finally? -> use the defaults for conversion to ImageDataBunch


# Now we can look at the created DataBunch:

data.show_batch(3, figsize=(6, 6), hide_axis=False)


# Let's look at another example from [`vision.data`](/vision.data.html#vision.data) with the planet dataset. This time, it's a multiclassification problem with the labels in a csv file and no given split between valid and train data, so we use a random split. The factory method is:

planet = untar_data(URLs.PLANET_TINY)
planet_tfms = get_transforms(flip_vert=True, max_lighting=0.1, max_zoom=1.05, max_warp=0.)


pd.read_csv(planet / "labels.csv").head()


data = ImageDataBunch.from_csv(planet, folder='train', size=128, suffix='.jpg', label_delim=' ', ds_tfms=planet_tfms)


# With the data block API we can rewrite this like that:

planet.ls()


pd.read_csv(planet / "labels.csv").head()


data = (ImageList.from_csv(planet, 'labels.csv', folder='train', suffix='.jpg')
        #Where to find the data? -> in planet 'train' folder
        .split_by_rand_pct()
        #How to split in train/valid? -> randomly with the default 20% in valid
        .label_from_df(label_delim=' ')
        #How to label? -> use the second column of the csv file and split the tags by ' '
        .transform(planet_tfms, size=128)
        #Data augmentation? -> use tfms with a size of 128
        .databunch())
        #Finally -> use the defaults for conversion to databunch


data.show_batch(rows=2, figsize=(9, 7))


# The data block API also allows you to get your data together in problems for which there is no direct [`ImageDataBunch`](/vision.data.html#ImageDataBunch) factory method. For a segmentation task, for instance, we can use it to quickly get a [`DataBunch`](/basic_data.html#DataBunch). Let's take the example of the [camvid dataset](http://mi.eng.cam.ac.uk/research/projects/VideoRec/CamVid/). The images are in an 'images' folder and their corresponding mask is in a 'labels' folder.

camvid = untar_data(URLs.CAMVID_TINY)
path_lbl = camvid / 'labels'
path_img = camvid / 'images'


# We have a file that gives us the names of the classes (what each code inside the masks corresponds to: a pedestrian, a tree, a road...)

codes = np.loadtxt(camvid / 'codes.txt', dtype=str); codes


# And we define the following function that infers the mask filename from the image filename.

get_y_fn = lambda x: path_lbl / f'{x.stem}_P{x.suffix}'


# Then we can easily define a [`DataBunch`](/basic_data.html#DataBunch) using the data block API. Here we need to use `tfm_y=True` in the transform call because we need the same transforms to be applied to the target mask as were applied to the image.

data = (SegmentationItemList.from_folder(path_img)
        #Where to find the data? -> in path_img and its subfolders
        .split_by_rand_pct()
        #How to split in train/valid? -> randomly with the default 20% in valid
        .label_from_func(get_y_fn, classes=codes)
        #How to label? -> use the label function on the file name of the data
        .transform(get_transforms(), tfm_y=True, size=128)
        #Data augmentation? -> use tfms with a size of 128, also transform the label images
        .databunch())
        #Finally -> use the defaults for conversion to databunch


data.show_batch(rows=2, figsize=(7, 5))


# Another example for object detection. We use our tiny sample of the [COCO dataset](http://cocodataset.org/#home) here. There is a helper function in the library that reads the annotation file and returns the list of images names with the list of labelled bboxes associated to it. We convert it to a dictionary that maps image names with their bboxes and then write the function that will give us the target for each image filename.

coco = untar_data(URLs.COCO_TINY)
images, lbl_bbox = get_annotations(coco / 'train.json')
img2bbox = dict(zip(images, lbl_bbox))
get_y_func = lambda o: img2bbox[o.name]


# The following code is very similar to what we saw before. The only new addition is the use of a special function to collate the samples in batches. This comes from the fact that our images may have multiple bounding boxes, so we need to pad them to the largest number of bounding boxes.

data = (ObjectItemList.from_folder(coco)
        #Where are the images? -> in coco and its subfolders
        .split_by_rand_pct()
        #How to split in train/valid? -> randomly with the default 20% in valid
        .label_from_func(get_y_func)
        #How to find the labels? -> use get_y_func on the file name of the data
        .transform(get_transforms(), tfm_y=True)
        #Data augmentation? -> Standard transforms; also transform the label images
        .databunch(bs=16, collate_fn=bb_pad_collate))
        #Finally we convert to a DataBunch, use a batch size of 16,
        # and we use bb_pad_collate to collate the data into a mini-batch


data.show_batch(rows=2, ds_type=DatasetType.Valid, figsize=(6, 6))


# But vision isn't the only application where the data block API works. It can also be used for text and tabular data. With our sample of the IMDB dataset (labelled texts in a csv file), here is how to get the data together for a language model.

from fastai.text import *


imdb = untar_data(URLs.IMDB_SAMPLE)


data_lm = (TextList
           .from_csv(imdb, 'texts.csv', cols='text')
           #Where are the text? Column 'text' of texts.csv
           .split_by_rand_pct()
           #How to split it? Randomly with the default 20% in valid
           .label_for_lm()
           #Label it for a language model
           .databunch())
           #Finally we convert to a DataBunch


data_lm.show_batch()


# For a classification problem, we just have to change the way labeling is done. Here we use the csv column `label`.

data_clas = (TextList.from_csv(imdb, 'texts.csv', cols='text')
                   .split_from_df(col='is_valid')
                   .label_from_df(cols='label')
                   .databunch())


data_clas.show_batch()


# Lastly, for tabular data, we just have to pass the name of our categorical and continuous variables as an extra argument. We also add some [`PreProcessor`](/data_block.html#PreProcessor)s that are going to be applied to our data once the splitting and labelling is done.

from fastai.tabular import *


adult = untar_data(URLs.ADULT_SAMPLE)
df = pd.read_csv(adult / 'adult.csv')
dep_var = 'salary'
cat_names = ['workclass', 'education', 'marital-status', 'occupation', 'relationship', 'race', 'sex', 'native-country']
cont_names = ['education-num', 'hours-per-week', 'age', 'capital-loss', 'fnlwgt', 'capital-gain']
procs = [FillMissing, Categorify, Normalize]


data = (TabularList.from_df(df, path=adult, cat_names=cat_names, cont_names=cont_names, procs=procs)
                           .split_by_idx(valid_idx=range(800, 1000))
                           .label_from_df(cols=dep_var)
                           .databunch())


data.show_batch()


# ## Step 1: Provide inputs

# The basic class to get your inputs into is the following one. It's also the same class that will contain all of your labels (hence the name [`ItemList`](/data_block.html#ItemList)).

show_doc(ItemList, title_level=3)


# This class regroups the inputs for our model in `items` and saves a `path` attribute which is where it will look for any files (image files, csv file with labels...). `label_cls` will be called to create the labels from the result of the label function, `inner_df` is an underlying dataframe, and `processor` is to be applied to the inputs after the splitting and labeling.

# It has multiple subclasses depending on the type of data you're handling. Here is a quick list:
#   - [`CategoryList`](/data_block.html#CategoryList) for labels in classification
#   - [`MultiCategoryList`](/data_block.html#MultiCategoryList) for labels in a multi classification problem
#   - [`FloatList`](/data_block.html#FloatList) for float labels in a regression problem
#   - [`ImageList`](/vision.data.html#ImageList) for data that are images
#   - [`SegmentationItemList`](/vision.data.html#SegmentationItemList) like [`ImageList`](/vision.data.html#ImageList) but will default labels to [`SegmentationLabelList`](/vision.data.html#SegmentationLabelList)
#   - [`SegmentationLabelList`](/vision.data.html#SegmentationLabelList) for segmentation masks
#   - [`ObjectItemList`](/vision.data.html#ObjectItemList) like [`ImageList`](/vision.data.html#ImageList) but will default labels to `ObjectLabelList`
#   - `ObjectLabelList` for object detection
#   - [`PointsItemList`](/vision.data.html#PointsItemList) for points (of the type [`ImagePoints`](/vision.image.html#ImagePoints))
#   - [`ImageImageList`](/vision.data.html#ImageImageList) for image to image tasks
#   - [`TextList`](/text.data.html#TextList) for text data
#   - [`TextList`](/text.data.html#TextList) for text data stored in files
#   - [`TabularList`](/tabular.data.html#TabularList) for tabular data
#   - [`CollabList`](/collab.html#CollabList) for collaborative filtering

# Once you have selected the class that is suitable, you can instantiate it with one of the following factory methods

show_doc(ItemList.from_folder)


path = untar_data(URLs.MNIST_TINY)
path.ls()


ImageList.from_folder(path)


show_doc(ItemList.from_df)


path = untar_data(URLs.MNIST_SAMPLE)
path.ls()


df = pd.read_csv(path / 'labels.csv')
df.head()


ImageList.from_df(df, path)


show_doc(ItemList.from_csv)


path = untar_data(URLs.MNIST_SAMPLE)
path.ls()


ImageList.from_csv(path, 'labels.csv')


# ### Optional step: filter your data

# The factory method may have grabbed too many items. For instance, if you were searching sub folders with the `from_folder` method, you may have gotten files you don't want. To remove those, you can use one of the following methods.

show_doc(ItemList.filter_by_func)


path = untar_data(URLs.MNIST_SAMPLE)
df = pd.read_csv(path / 'labels.csv')
df.head()


# Suppose that you only want to keep images with a suffix ".png". Well, this method will do magic for you.

Path(df.name[0]).suffix


ImageList.from_df(df, path).filter_by_func(lambda fname: Path(fname).suffix == '.png')


show_doc(ItemList.filter_by_folder)


show_doc(ItemList.filter_by_rand)


path = untar_data(URLs.MNIST_SAMPLE)
ImageList.from_folder(path).filter_by_rand(0.5)


# Contrast the number of items with the list created without the filter.

ImageList.from_folder(path)


show_doc(ItemList.to_text)


path = untar_data(URLs.MNIST_SAMPLE)
pd.read_csv(path / 'labels.csv').head()


file_name = "item_list.txt"
ImageList.from_folder(path).to_text(file_name)


get_ipython().system(' cat {path/file_name} | head')


show_doc(ItemList.use_partial_data)


path = untar_data(URLs.MNIST_SAMPLE)
ImageList.from_folder(path).use_partial_data(0.5)


# Contrast the number of items with the list created without the filter.

ImageList.from_folder(path)


# ### Writing your own [`ItemList`](/data_block.html#ItemList)

# First check if you can't easily customize one of the existing subclass by:
# - subclassing an existing one and replacing the `get` method (or the `open` method if you're dealing with images)
# - applying a custom `processor` (see step 4)
# - changing the default `label_cls` for the label creation
# - adding a default [`PreProcessor`](/data_block.html#PreProcessor) with the `_processor` class variable
#
# If this isn't the case and you really need to write your own class, there is a [full tutorial](/tutorial.itemlist) that explains how to proceed.

show_doc(ItemList.analyze_pred)


show_doc(ItemList.get)


show_doc(ItemList.new)


# You'll never need to subclass this normally, just don't forget to add to `self.copy_new` the names of the arguments that needs to be copied each time `new` is called in `__init__`.

show_doc(ItemList.reconstruct)


# ## Step 2: Split the data between the training and the validation set

# This step is normally straightforward, you just have to pick oe of the following functions depending on what you need.

show_doc(ItemList.split_none)


show_doc(ItemList.split_by_rand_pct)


show_doc(ItemList.split_subsets)


# This function is handy if you want to work with subsets of specific sizes, e.g., you want to use 20% of the data for the validation dataset, but you only want to train on a small subset of the rest of the data: `split_subsets(train_size=0.08, valid_size=0.2)`.

show_doc(ItemList.split_by_files)


show_doc(ItemList.split_by_fname_file)


show_doc(ItemList.split_by_folder)


jekyll_note("This method looks at the folder immediately after `self.path` for `valid` and `train`.")


show_doc(ItemList.split_by_idx)


show_doc(ItemList.split_by_idxs)


show_doc(ItemList.split_by_list)


show_doc(ItemList.split_by_valid_func)


show_doc(ItemList.split_from_df)


jekyll_warn("This method assumes the data has been created from a csv file or a dataframe.")


# ## Step 3: Label the inputs

# To label your inputs, use one of the following functions. Note that even if it's not in the documented arguments, you can always pass a `label_cls` that will be used to create those labels (the default is the one from your input [`ItemList`](/data_block.html#ItemList), and if there is none, it will go to [`CategoryList`](/data_block.html#CategoryList),  [`MultiCategoryList`](/data_block.html#MultiCategoryList) or [`FloatList`](/data_block.html#FloatList) depending on the type of the labels). This is implemented in the following function:

show_doc(ItemList.get_label_cls)


# If no `label_cls` argument is passed, the correct labeling type can usually be inferred based on the data (for classification or regression). If you have multiple regression targets (e.g. predict 5 different numbers from a single image/text), be aware that arrays of floats are by default considered to be targets for one-hot encoded classification. If your task is regression, be sure the pass `label_cls = FloatList` so that learners created from your databunch initialize correctly.

# The first example in these docs created labels as follows:

path = untar_data(URLs.MNIST_TINY)
ll = ImageList.from_folder(path).split_by_folder().label_from_folder().train


# If you want to save the data necessary to recreate your [`LabelList`](/data_block.html#LabelList) (not including saving the actual image/text/etc files), you can use `to_df` or `to_csv`:
#
# ```python
# ll.train.to_csv('tmp.csv')
# ```
#
# Or just grab a `pd.DataFrame` directly:

ll.to_df().head()


show_doc(ItemList.label_empty)


show_doc(ItemList.label_from_df)


jekyll_warn("This method only works with data objects created with either `from_csv` or `from_df` methods.")


show_doc(ItemList.label_const)


show_doc(ItemList.label_from_folder)


jekyll_note("This method looks at the last subfolder in the path to determine the classes.")


show_doc(ItemList.label_from_func)


show_doc(ItemList.label_from_re)


show_doc(CategoryList, title_level=3)


# [`ItemList`](/data_block.html#ItemList) suitable for storing labels in `items` belonging to `classes`. If `None` are passed, `classes` will be determined by the unique different labels. `processor` will default to [`CategoryProcessor`](/data_block.html#CategoryProcessor).

show_doc(MultiCategoryList, title_level=3)


# It will store list of labels in `items` belonging to `classes`. If `None` are passed, `classes` will be determined by the unique different labels. `sep` is used to split the content of `items` in a list of tags.
#
# If `one_hot=True`, the items contain the labels one-hot encoded. In this case, it is mandatory to pass a list of `classes` (as we can't use the different labels).

show_doc(FloatList, title_level=3)


show_doc(EmptyLabelList, title_level=3)


# ## Invisible step: preprocessing

# This isn't seen here in the API, but if you passed a `processor` (or a list of them) in your initial [`ItemList`](/data_block.html#ItemList) during step 1, it will be applied here. If you didn't pass any processor, a list of them might still be created depending on what is in the `_processor` variable of your class of items (this can be a list of [`PreProcessor`](/data_block.html#PreProcessor) classes).
#
# A processor is a transformation that is applied to all the inputs once at initialization, with a state computed on the training set that is then applied without modification on the validation set (and maybe the test set). For instance, it can be processing texts to tokenize then numericalize them. In that case we want the validation set to be numericalized with exactly the same vocabulary as the training set.
#
# Another example is in tabular data, where we fill missing values with (for instance) the median computed on the training set. That statistic is stored in the inner state of the [`PreProcessor`](/data_block.html#PreProcessor) and applied on the validation set.
#
# This is the generic class for all processors.

show_doc(PreProcessor, title_level=3)


show_doc(PreProcessor.process_one)


# Process one `item`. This method needs to be written in any subclass.

show_doc(PreProcessor.process)


# Process a dataset. This default to apply `process_one` on every `item` of `ds`.

show_doc(CategoryProcessor, title_level=3)


show_doc(CategoryProcessor.generate_classes)


show_doc(MultiCategoryProcessor, title_level=3)


show_doc(MultiCategoryProcessor.generate_classes)


# ## Optional steps

# ### Add transforms

# Transforms differ from processors in the sense they are applied on the fly when we grab one item. They also may change each time we ask for the same item in the case of random transforms.

show_doc(LabelLists.transform)


# This is primary for the vision application. The `kwargs` arguments are the ones expected by the type of transforms you pass. `tfm_y` is among them and if set to `True`, the transforms will be applied to input and target.
#
# For examples see: [vision.transforms](vision.transform.html).

# ### Add a test set

# To add a test set, you can use one of the two following methods.

show_doc(LabelLists.add_test)


jekyll_note("Here `items` can be an `ItemList` or a collection.")


show_doc(LabelLists.add_test_folder)


jekyll_warn("In fastai the test set is unlabeled! No labels will be collected even if they are available.")


# Instead, either the passed `label` argument or an empty label will be used for all entries of this dataset (this is required by the internal pipeline of fastai).
#
# In the `fastai` framework `test` datasets have no labels - this is the unknown data to be predicted. If you want to validate your model on a `test` dataset with labels, you probably need to use it as a validation set, as in:
#
# ```
# data_test = (ImageList.from_folder(path)
#         .split_by_folder(train='train', valid='test')
#         .label_from_folder()
#         ...)
# ```
#
# Another approach, where you do use a normal validation set, and then when the training is over, you just want to validate the test set w/ labels as a validation set, you can do this:
#
# ```
# tfms = []
# path = Path('data').resolve()
# data = (ImageList.from_folder(path)
#         .split_by_pct()
#         .label_from_folder()
#         .transform(tfms)
#         .databunch()
#         .normalize() )
# learn = cnn_learner(data, models.resnet50, metrics=accuracy)
# learn.fit_one_cycle(5,1e-2)
#
# # now replace the validation dataset entry with the test dataset as a new validation dataset:
# # everything is exactly the same, except replacing `split_by_pct` w/ `split_by_folder`
# # (or perhaps you were already using the latter, so simply switch to valid='test')
# data_test = (ImageList.from_folder(path)
#         .split_by_folder(train='train', valid='test')
#         .label_from_folder()
#         .transform(tfms)
#         .databunch()
#         .normalize()
#        )
# learn.validate(data_test.valid_dl)
# ```
# Of course, your data block can be totally different, this is just an example.

# ## Step 4: convert to a [`DataBunch`](/basic_data.html#DataBunch)

# This last step is usually pretty straightforward. You just have to include all the arguments we pass to [`DataBunch.create`](/basic_data.html#DataBunch.create) (`bs`, `num_workers`,  `collate_fn`). The class called to create a [`DataBunch`](/basic_data.html#DataBunch) is set in the `_bunch` attribute of the inputs of the training set if you need to modify it. Normally, the various subclasses we showed before handle that for you.

show_doc(LabelLists.databunch)


# ## Inner classes

show_doc(LabelList, title_level=3)


# Optionally apply `tfms` to `y` if `tfm_y` is `True`.

show_doc(LabelList.export)


show_doc(LabelList.transform_y)


show_doc(LabelList.get_state)


show_doc(LabelList.load_empty)


show_doc(LabelList.load_state)


show_doc(LabelList.process)


show_doc(LabelList.set_item)


show_doc(LabelList.to_df)


show_doc(LabelList.to_csv)


show_doc(LabelList.transform)


show_doc(ItemLists, title_level=3)


show_doc(ItemLists.label_from_lists)


show_doc(ItemLists.transform)


show_doc(ItemLists.transform_y)


show_doc(LabelLists, title_level=3)


show_doc(LabelLists.get_processors)


show_doc(LabelLists.load_empty)


show_doc(LabelLists.load_state)


show_doc(LabelLists.process)


# ## Helper functions

show_doc(get_files)


# ## Undocumented Methods - Methods moved below this line will intentionally be hidden

show_doc(CategoryList.new)


show_doc(LabelList.new)


show_doc(CategoryList.get)


show_doc(LabelList.predict)


show_doc(ItemList.new)


show_doc(ItemList.process_one)


show_doc(ItemList.process)


show_doc(MultiCategoryProcessor.process_one)


show_doc(FloatList.get)


show_doc(CategoryProcessor.process_one)


show_doc(CategoryProcessor.create_classes)


show_doc(CategoryProcessor.process)


show_doc(MultiCategoryList.get)


show_doc(FloatList.new)


show_doc(FloatList.reconstruct)


show_doc(MultiCategoryList.analyze_pred)


show_doc(MultiCategoryList.reconstruct)


show_doc(CategoryList.reconstruct)


show_doc(CategoryList.analyze_pred)


show_doc(EmptyLabelList.reconstruct)


show_doc(EmptyLabelList.get)


show_doc(LabelList.databunch)


# ## New Methods - Please document or move to the undocumented section

show_doc(ItemList.add)
