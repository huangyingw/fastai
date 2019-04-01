
# coding: utf-8

# ## Functional Test Documentation

# Generates documentation for fastai's functional tests

from fastai.gen_doc.nbdoc import *
from fastai.gen_doc.nbtest import *


# ## Find tests for any function/class

# [`show_test`](/gen_doc.nbtest.html#show_test) and [`doctest`](/gen_doc.nbtest.html#doctest) searches for any implemented tests for a given fastai class or function
#
# For test writers:
# * Use this module to search for tests and get a better idea on which parts of the fastai api need more functional tests
#
# For fastai users:
# * Usage is similar to [`nbdoc.show_doc`](/gen_doc.nbdoc.html#show_doc) and [`nbdoc.doc`](/gen_doc.nbdoc.html#doc).
# * It's here to help you find associated tests for a given function can help understand usage.
#

# ## Usage:

show_doc(show_test)


# **Show tests from function**

from fastai.basic_train import Learner
show_test(Learner.fit)


# **Show tests from a Class**

from fastai.basic_data import DataBunch
show_test(DataBunch)


from fastai.text.data import TextList
show_test(TextList)


# ## Different test types

# Above, you will see 2 different test types: `Tests found for...` and `Some other tests...`
#
# * `Tests found for...` - Searches for function matches in `test_registry.json`. This json file is populated from `doctest.this_tests` calls.
# * `Some other tests...` - Returns any test function where the fastai function in called inside the body

# ## Show in notebook inline:

show_doc(doctest)


# ## Internal search methods

show_doc(lookup_db)


show_doc(find_related_tests)


show_doc(find_test_matches)


show_doc(find_test_files)


show_doc(fuzzy_test_match)


# ## Undocumented Methods - Methods moved below this line will intentionally be hidden
