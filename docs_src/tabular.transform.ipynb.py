
# coding: utf-8

# ## Tabular data preprocessing

from fastai.gen_doc.nbdoc import *
from fastai.tabular import *


# ## Overview

# This package contains the basic class to define a transformation for preprocessing dataframes of tabular data, as well as basic [`TabularProc`](/tabular.transform.html#TabularProc). Preprocessing includes things like
# - replacing non-numerical variables by categories, then their ids,
# - filling missing values,
# - normalizing continuous variables.
#
# In all those steps we have to be careful to use the correspondence we decide on our training set (which id we give to each category, what is the value we put for missing data, or how the mean/std we use to normalize) on our validation or test set. To deal with this, we use a special class called [`TabularProc`](/tabular.transform.html#TabularProc).
#
# The data used in this document page is a subset of the [adult dataset](https://archive.ics.uci.edu/ml/datasets/adult). It gives a certain amount of data on individuals to train a model to predict whether their salary is greater than \$50k or not.

path = untar_data(URLs.ADULT_SAMPLE)
df = pd.read_csv(path / 'adult.csv')
train_df, valid_df = df.iloc[:800].copy(), df.iloc[800:1000].copy()
train_df.head()


# We see it contains numerical variables (like `age` or `education-num`) as well as categorical ones (like `workclass` or `relationship`). The original dataset is clean, but we removed a few values to give examples of dealing with missing variables.

cat_names = ['workclass', 'education', 'marital-status', 'occupation', 'relationship', 'race', 'sex', 'native-country']
cont_names = ['age', 'fnlwgt', 'education-num', 'capital-gain', 'capital-loss', 'hours-per-week']


# ## Transforms for tabular data

show_doc(TabularProc)


# Base class for creating transforms for dataframes with categorical variables `cat_names` and continuous variables `cont_names`. Note that any column not in one of those lists won't be touched.

show_doc(TabularProc.__call__)


show_doc(TabularProc.apply_train)


show_doc(TabularProc.apply_test)


jekyll_important("Those two functions must be implemented in a subclass. `apply_test` defaults to `apply_train`.")


# The following [`TabularProc`](/tabular.transform.html#TabularProc) are implemented in the fastai library. Note that the replacement from categories to codes as well as the normalization of continuous variables are automatically done in a [`TabularDataBunch`](/tabular.data.html#TabularDataBunch).

show_doc(Categorify)


# Variables in `cont_names` aren't affected.

show_doc(Categorify.apply_train)


show_doc(Categorify.apply_test)


tfm = Categorify(cat_names, cont_names)
tfm(train_df)
tfm(valid_df, test=True)


# Since we haven't changed the categories by their codes, nothing visible has changed in the dataframe yet, but we can check that the variables are now categorical and view their corresponding codes.

train_df['workclass'].cat.categories


# The test set will be given the same category codes as the training set.

valid_df['workclass'].cat.categories


show_doc(FillMissing)


# `cat_names` variables are left untouched (their missing value will be replaced by code 0 in the [`TabularDataBunch`](/tabular.data.html#TabularDataBunch)). [`fill_strategy`](#FillStrategy) is adopted to replace those nans and if `add_col` is True, whenever a column `c` has missing values, a column named `c_nan` is added and flags the line where the value was missing.

show_doc(FillMissing.apply_train)


show_doc(FillMissing.apply_test)


# Fills the missing values in the `cont_names` columns with the ones picked during train.

train_df[cont_names].head()


tfm = FillMissing(cat_names, cont_names)
tfm(train_df)
tfm(valid_df, test=True)
train_df[cont_names].head()


# Values missing in the `education-num` column are replaced by 10, which is the median of the column in `train_df`. Categorical variables are not changed, since `nan` is simply used as another category.

valid_df[cont_names].head()


show_doc(FillStrategy, alt_doc_string='Enum flag represents determines how `FillMissing` should handle missing/nan values', arg_comments={
    'MEDIAN': 'nans are replaced by the median value of the column',
    'COMMON': 'nans are replaced by the most common value of the column',
    'CONSTANT': 'nans are replaced by `fill_val`'
})


show_doc(Normalize)


show_doc(Normalize.apply_train)


show_doc(Normalize.apply_test)


# ## Treating date columns

show_doc(add_datepart)


# Will `drop` the column in `df` if the flag is `True`. The `time` flag decides if we go down to the time parts or stick to the date parts.

# ## Splitting data into cat and cont

show_doc(cont_cat_split)


# Parameters:
# - df: A pandas data frame.
# - max_card: Maximum cardinality of a numerical categorical variable.
# - dep_var: A dependent variable.
#
# Return:
# - cont_names: A list of names of continuous variables.
# - cat_names: A list of names of categorical variables.

df = pd.DataFrame({'col1': [1, 2, 3], 'col2': ['a', 'b', 'a'], 'col3': [0.5, 1.2, 7.5], 'col4': ['ab', 'o', 'o']})
df


cont_list, cat_list = cont_cat_split(df=df, max_card=20, dep_var='col4')
cont_list, cat_list


# ## Undocumented Methods - Methods moved below this line will intentionally be hidden

# ## New Methods - Please document or move to the undocumented section
