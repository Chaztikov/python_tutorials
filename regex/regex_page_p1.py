# #..//developers.google.c#../e#../pyth#../regular-expressions
import sys, os, re, pickle, subprocess
print(sys.version)

import matplotlib.pyplot as plt
import matplotlib as mpl
mpl.rcParams['font.serif']='Times'
mpl.rcParams['font.size']= 10.0
mpl.rcParams['grid.alpha']= 1.0
mpl.rcParams['grid.color']= '#b0b0b0'
mpl.rcParams['grid.linestyle']= '-'
mpl.rcParams['grid.linewidth']= 1.8
mpl.rcParams['lines.linewidth']= 2.5
mpl.rcParams['lines.marker']= 'None'
mpl.rcParams['figure.dpi']= 100.0

import numpy as np
import pandas as pd

import os;import numpy as np;import pandas as pd
import os,sys,re,subprocess
import numpy as np
import scipy
import seaborn as sns

import pandas as pd
import pandas_datareader.data as web
import matplotlib.pyplot as plt
import matplotlib as mpl
from matplotlib import style
import datetime

import scipy.integrate
from scipy.spatial import KDTree
from scipy.interpolate import BSpline
from scipy.interpolate import splrep, splder, sproot,make_interp_spline
import scipy.sparse.linalg as spla


df1 = pd.DataFrame({'A': ['A0', 'A1', 'A2', 'A3'],
'B': ['B0', 'B1', 'B2', 'B3'],
'C': ['C0', 'C1', 'C2', 'C3'],
'D': ['D0', 'D1', 'D2', 'D3']},
                    index=[0, 1, 2, 3])
 


df2 = pd.DataFrame({ 'A': ['A4', 'A5', 'A6', 'A7'],
                      'B': ['B4', 'B5', 'B6', 'B7'],
                      'C': ['C4', 'C5', 'C6', 'C7'],
                      'D': ['D4', 'D5', 'D6', 'D7']},
                    index=[4, 5, 6, 7])
 


df3 = pd.DataFrame({ 'A': ['A8', 'A9', 'A10', 'A11'],
                      'B': ['B8', 'B9', 'B10', 'B11'],
                      'C': ['C8', 'C9', 'C10', 'C11'],
                      'D': ['D8', 'D9', 'D10', 'D11']},
                    index=[8, 9, 10, 11])
 


frames = [df1, df2, df3]


result = pd.concat(frames)
# pd.concat(objs, axis=0, join='outer', join_axes=None, ignore_index=False,
#        keys=None, levels=None, names=None, verify_integrity=False,
#        copy=True)
# ob a sequence or mapping of Series, DataFrame, or Panel objects. If a dict is passed, the sorted keys will be used as the keys argument, unless it is passed, in which case the values will be selected (see below). Any None objects will be dropped silently unless they are all None in which case a ValueError will be raised.
# ax {0, 1, …}, default 0. The axis to concatenate along.
# jo {‘inner’, ‘outer’}, default ‘outer’. How to handle indexes on other axis(es). Outer for union and inner for intersection.
# ignore_ind boolean, default False. If True, do not use the index values on the concatenation axis. The resulting axis will be labeled 0, …, n - 1. This is useful if you are concatenating objects where the concatenation axis does not have meaningful indexing information. Note the index values on the other axes are still respected in the join.
# join_ax list of Index objects. Specific indexes to use for the other n - 1 axes instead of performing inn#../outer set logic.
# ke sequence, default None. Construct hierarchical index using the passed keys as the outermost level. If multiple levels passed, should contain tuples.
# leve list of sequences, default None. Specific levels (unique values) to use for constructing a MultiIndex. Otherwise they will be inferred from the keys.
# nam list, default None. Names for the levels in the resulting hierarchical index.
# verify_integri boolean, default False. Check whether the new concatenated axis contains duplicates. This can be very expensive relative to the actual data concatenation.
# co boolean, default True. If False, do not copy data unnecessarily.
# Without a little bit of context many of these arguments don’t make much sense. Let’s revisit the above example. Suppose we wanted to associate specific keys with each of the pieces of the chopped up DataFrame. We can do this using the keys argum

'''
result = pd.concat(frames, keys=['x', 'y', 'z'])

# #../_imag#../merging_concat_keys.png
# As you can see (if you’ve read the rest of the documentation), the resulting object’s index has a hierarchical index. This means that we can now select out each chunk b  


result.loc['y']
'''

   #           D
   # A   B   C   D4
   # A   B   C   D5
   # A   B   C   D6
   # A   B   C   D7
'''
It’s not a stretch to see how this can be very useful. More detail on this functionality below.

Note It is worth noting that concat() (and therefore append()) makes a full copy of the data, and that constantly reusing this function can create a significant performance hit. If you need to use the operation over several datasets, use a list comprehension.
frames = [ process_your_file(f) for f in files ]
result = pd.concat(frames)
Set logic on the other axes
When gluing together multiple DataFrames, you have a choice of how to handle the other axes (other than the one being concatenated). This can be done in the following three w

Take the union of them all, join='outer'. This is the default option as it results in zero information loss.
Take the intersection, join='inner'.
Use a specific index, as passed to the join_axes argument.
Here is an example of each of these methods. First, the default join='outer' behav

'''
df4 = pd.DataFrame({ 'B':['B2', 'B3', 'B6', 'B7'],
                      'D':['D2', 'D3', 'D6', 'D7'],
                      'F':['F2', 'F3', 'F6', 'F7']},
                    index=[2, 3, 6, 7])
 

result = pd.concat([df1, df4], axis=1)#)#, sort=False)
'''

#../_imag#../merging_concat_axis1.png
Warning
Changed in version 0.23.0.

The default behavior with join='outer' is to sort the other axis (columns in this case). In a future version of pandas, the default will be to not sort. We specified sort=False to opt in to the new behavior now.

Here is the same thing with join='inn

'''
result = pd.concat([df1, df4], axis=1, join='inner')

# #../_imag#../merging_concat_axis1_inner.png
# Lastly, suppose we just wanted to reuse the exact index from the original DataFr

result = pd.concat([df1, df4], axis=1, join_axes=[df1.index])
'''

#../_imag#../merging_concat_axis1_join_axes.png
Concatenating using append
A useful shortcut to concat() are the append() instance methods on Series and DataFrame. These methods actually predated concat. They concatenate along axis=0, namely the in

'''
result = df1.append(df2)

# #../_imag#../merging_append1.png
# In the case of DataFrame, the indexes must be disjoint but the columns do not need to

result = df1.append(df4)#, sort=False)
'''

#../_imag#../merging_append2.png
append may take multiple objects to concaten

'''
result = df1.append([df2, df3])
'''
#../_imag#../merging_append3.png
Note Unlike the append() method, which appends to the original list and returns None, append() here does not modify df1 and returns its copy with df2 appended.
Ignoring indexes on the concatenation axis
For DataFrame objects which don’t have a meaningful index, you may wish to append them and ignore the fact that they may have overlapping indexes. To do this, use the ignore_index argum

'''
result = pd.concat([df1, df4], ignore_index=True)#, sort=False)

#../_imag#../merging_concat_ignore_index.png
# This is also a valid argument to DataFrame.appen

result = df1.append(df4, ignore_index=True)#, sort=False)
'''

#../_imag#../merging_append_ignore_index.png
Concatenating with mixed ndims
You can concatenate a mix of Series and DataFrame objects. The Series will be transformed to DataFrame with the column name as the name of the Series.

'''
s1 = pd.Series(['X0', 'X1', 'X2', 'X3'], name='X')

result = pd.concat([df1, s1], axis=1)
'''

#../_imag#../merging_concat_mixed_ndim.png
Note Since we’re concatenating a Series to a DataFrame, we could have achieved the same result with DataFrame.assign(). To concatenate an arbitrary number of pandas objects (DataFrame or Series), use concat.
If unnamed Series are passed they will be numbered consecutively.

'''
s2 = pd.Series(['_0', '_1', '_2', '_3'])

result = pd.concat([df1, s2, s2, s2], axis=1)
'''

#../_imag#../merging_concat_unnamed_series.png
Passing ignore_index=True will drop all name references.

'''
result = pd.concat([df1, s1], axis=1, ignore_index=True)

#../_imag#../merging_concat_series_ignore_index.png
# More concatenating with group keys
# A fairly common use of the keys argument is to override the column names when creating a new DataFrame based on existing Series. Notice how the default behaviour consists on letting the resulting DataFrame inherit the parent Series’ name, when these existed.

s3 = pd.Series([0, 1, 2, 3], name='foo')
'''

'''
s4 = pd.Series([0, 1, 2, 3])

s5 = pd.Series([0, 1, 4, 5])
'''

'''
pd.concat([s3, s4, s5], axis=1)
'''

fo      1
           0
           1
           4
           5
Through the keys argument we can override the existing column names.

'''
pd.concat([s3, s4, s5], axis=1, keys=['red', 'blue', 'yellow'])
'''

re   blu   yellow
                   0
                   1
                   4
                   5
Let’s consider a variation of the very first example presen

'''
result = pd.concat(frames, keys=['x', 'y', 'z'])

# #../_imag#../merging_concat_group_keys2.png
# You can also pass a dict to concat in which case the dict keys will be used for the keys argument (unless other keys are specifi

pieces = [df1,df2,df3]
'''

'''
result = pd.concat(pieces)

# #../_imag#../merging_concat_dict.png
'''
result = pd.concat(pieces, keys=['z', 'y'])

#../_imag#../merging_concat_dict_keys.png
The MultiIndex created has levels that are constructed from the passed keys and the index of the DataFrame pie

'''
result.index.levels
'''
FrozenList([['z', 'y'], [4, 5, 6, 7, 8, 9, 10, 11]])
If you wish to specify other levels (as will occasionally be the case), you can do so using the levels argum

'''
result = pd.concat(pieces, keys=['x', 'y', 'z'],
                     levels=[['z', 'y', 'x', 'w']],
                     names=['group_key'])
  

# #../_imag#../merging_concat_dict_keys_names.png
'''
result.index.levels
'''
FrozenList([['z', 'y', 'x', 'w'], [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11]])
# This is fairly esoteric, but it is actually necessary for implementing things like GroupBy where the order of a categorical variable is meaningful.

# Appending rows to a DataFrame
# While not especially efficient (since a new object must be created), you can append a single row to a DataFrame by passing a Series or dict to append, which returns a new DataFrame as above.

'''
s2 = pd.Series(['X0', 'X1', 'X2', 'X3'], index=['A', 'B', 'C', 'D'])

'''
result = df1.append(s2, ignore_index=True)

# #../_imag#../merging_append_series_as_row.png
# You should use ignore_index with this method to instruct DataFrame to discard its index. If you wish to preserve the index, you should construct an appropriately-indexed DataFrame and append or concatenate those objects.

# You can also pass a list of dicts or Ser

dicts = [{ 1,2,3,4},{ 5,6,7,8}]
  
'''

'''
result = df1.append(dicts, ignore_index=True)#, sort=False)
'''
#../_imag#../merging_append_dits.png
Database-style DataFrame or named Series joini#../merging
pandas has full-featured, high performance in-memory join operations idiomatically very similar to relational databases like SQL. These methods perform significantly better (in some cases well over an order of magnitude better) than other open source implementations (like b:merge.data.frame in R). The reason for this is careful algorithmic design and the internal layout of the data in DataFrame.

See the cookbook for some advanced strategies.

Users who are familiar with SQL but new to pandas might be interested in a comparison with SQL.

pandas provides a single function, merge(), as the entry point for all standard database join operations between DataFrame or named Series obje
'''
pd.merge(left, right, how='inner', on=None, left_on=None, right_on=None,
      left_index=False, right_index=False, sort=True,
      suffixes=('_x', '_y'), copy=True, indicator=False,
      validate=None)
'''
l A DataFrame or named Series object.

ri Another DataFrame or named Series object.

on: Column or index level names to join on. Must be found in both the left and right DataFrame a#../or Series objects. If not passed and left_index and right_index are False, the intersection of the columns in the DataFrames a#../or Series will be inferred to be the join keys.

left Columns or index levels from the left DataFrame or Series to use as keys. Can either be column names, index level names, or arrays with length equal to the length of the DataFrame or Series.

right Columns or index levels from the right DataFrame or Series to use as keys. Can either be column names, index level names, or arrays with length equal to the length of the DataFrame or Series.

left_in If True, use the index (row labels) from the left DataFrame or Series as its join key(s). In the case of a DataFrame or Series with a MultiIndex (hierarchical), the number of levels must match the number of join keys from the right DataFrame or Series.

right_in Same usage as left_index for the right DataFrame or Series

One of 'left', 'right', 'outer', 'inner'. Defaults to inner. See below for more detailed description of each method.

s Sort the result DataFrame by the join keys in lexicographical order. Defaults to True, setting to False will improve performance substantially in many cases.

suffi A tuple of string suffixes to apply to overlapping columns. Defaults to ('_x', '_y').

c Always copy data (default True) from the passed DataFrame or named Series objects, even when reindexing is not necessary. Cannot be avoided in many cases but may improve performanc#../ memory usage. The cases where copying can be avoided are somewhat pathological but this option is provided nonetheless.

indica Add a column to the output DataFrame called _merge with information on the source of each row. _merge is Categorical-type and takes on a value of left_only for observations whose merge key only appears in 'left' DataFrame or Series, right_only for observations whose merge key only appears in 'right' DataFrame or Series, and both if the observation’s merge key is found in both.

valida string, default None. If specified, checks if merge is of specified type.

“one_to_one” or1”: checks if merge keys are unique in both left and right datasets.
“one_to_many” orm”: checks if merge keys are unique in left dataset.
“many_to_one” or1”: checks if merge keys are unique in right dataset.
“many_to_many” orm”: allowed, but does not result in checks.
New in version 0.21.0.

Note Support for specifying index levels as the on, left_on, and right_on parameters was added in version 0.23.0. Support for merging named Series objects was added in version 0.24.0.
The return type will be the same as left. If left is a DataFrame or named Series and right is a subclass of DataFrame, the return type will still be DataFrame.

merge is a function in the pandas namespace, and it is also available as a DataFrame instance method merge(), with the calling DataFrame being implicitly considered the left object in the join.

The related join() method, uses merge internally for the index-on-index (by default) and column(s)-on-index join. If you are joining on index only, you may wish to use DataFrame.join to save yourself some typing.

Brief primer on merge methods (relational algebra)
Experienced users of relational databases like SQL will be familiar with the terminology used to describe join operations between two SQL-table like structures (DataFrame objects). There are several cases to consider which are very important to underst

one-to-one jo for example when joining two DataFrame objects on their indexes (which must contain unique values).
many-to-one jo for example when joining an index (unique) to one or more columns in a different DataFrame.
many-to-many jo joining columns on columns.
Note When joining columns on columns (potentially a many-to-many join), any indexes on the passed DataFrame objects will be discarded.
It is worth spending some time understanding the result of the many-to-many join case. In SQ#../ standard relational algebra, if a key combination appears more than once in both tables, the resulting table will have the Cartesian product of the associated data. Here is a very basic example with one unique key combinat

'''
left = pd.DataFrame({ ['K0', 'K1', 'K2', 'K3'],
                        ['A0', 'A1', 'A2', 'A3'],
                        ['B0', 'B1', 'B2', 'B3']})
  

right = pd.DataFrame({ ['K0', 'K1', 'K2', 'K3'],
                         ['C0', 'C1', 'C2', 'C3'],
                         ['D0', 'D1', 'D2', 'D3']})
  
'''

'''
result = pd.merge(left, right, on='key')

# #../_imag#../merging_merge_on_key.png
# Here is a more complicated example with multiple join keys. Only the keys appearing in left and right are present (the intersection), since how='inner' by default.

'''
left = pd.DataFrame({['K0', 'K0', 'K1', 'K2'],
                       ['K0', 'K1', 'K0', 'K1'],
                        ['A0', 'A1', 'A2', 'A3'],
                        ['B0', 'B1', 'B2', 'B3']})
  

'''
right = pd.DataFrame({['K0', 'K1', 'K1', 'K2'],
                        ['K0', 'K0', 'K0', 'K0'],
                         ['C0', 'C1', 'C2', 'C3'],
                         ['D0', 'D1', 'D2', 'D3']})
  

result = pd.merge(left, right, on=['key1', 'key2'])
'''

#../_imag#../merging_merge_on_key_multiple.png
The how argument to merge specifies how to determine which keys are to be included in the resulting table. If a key combination does not appear in either the left or right tables, the values in the joined table will be NA. Here is a summary of the how options and their SQL equivalent na

Merge method	SQL Join Name	Description
left	LEFT OUTER JOIN	Use keys from left frame only
right	RIGHT OUTER JOIN	Use keys from right frame only
outer	FULL OUTER JOIN	Use union of keys from both frames
inner	INNER JOIN	Use intersection of keys from both frames
'''
result = pd.merge(left, right, how='left', on=['key1', 'key2'])

#../_imag#../merging_merge_on_key_left.png
'''
result = pd.merge(left, right, how='right', on=['key1', 'key2'])

#../_imag#../merging_merge_on_key_right.png
'''
result = pd.merge(left, right, how='outer', on=['key1', 'key2'])

#../_imag#../merging_merge_on_key_outer.png

result = pd.merge(left, right, how='inner', on=['key1', 'key2'])


#../_imag#../merging_merge_on_key_inner.png
# Here is another example with duplicate join keys in DataFra

'''
left = pd.DataFrame({ [1, 2]   [2, 2]})

'''
right = pd.DataFrame({ [4, 5, 6]   [2, 2, 2]})

result = pd.merge(left, right, on='B', how='outer')
