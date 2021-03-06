
Users can use the validate argument to automatically check whether there are unexpected duplicates in their merge keys. Key uniqueness is checked before merge operations and so should protect against memory overflows. Checking key uniqueness is also a good way to ensure user data structures are as expected.

In the following example, there are duplicate values of B in the right DataFrame. As this is not a one-to-one merge – as specified in the validate argument – an exception will be raised.

'''
 left = pd.DataFrame({' [1,2], ' [1, 2]})

'''
 right = pd.DataFrame({' [4,5,6],  [2, 2, 2]})
'''
 result = pd.merge(left, right, on='B', how='outer', validate="one_to_one")
...
# MergeEr Merge keys are not unique in right dataset; not a one-to-one merge
# If the user is aware of the duplicates in the right DataFrame but wants to ensure there are no duplicates in the left DataFrame, one can use the validate='one_to_many' argument instead, which will not raise an exception.

'''
 pd.merge(left, right, on='B', how='outer', validate="one_to_many")
'''
 
#    A_x  B  A_y
# 0    1  1  NaN
# 1    2  2  4.0
# 2    2  2  5.0
# 3    2  2  6.0
# The merge indicator
# merge() accepts the argument indicator. If True, a Categorical-type column called _merge will be added to the output object that takes on val

# Observation Origin	_merge value
# Merge key only in 'left' frame	left_only
# Merge key only in 'right' frame	right_only
# Merge key in both frames	both
'''
 df1 = pd.DataFrame({'co [0, 1], 'col_le ['a', 'b']})

'''
 df2 = pd.DataFrame({'co [1, 2, 2], 'col_rig [2, 2, 2]})

'''
 pd.merge(df1, df2, on='col1', how='outer', indicator=True)
'''
 
   col1 col_left  col_right      _merge
0     0        a        NaN   left_only
1     1        b        2.0        both
2     2      NaN        2.0  right_only
3     2      NaN        2.0  right_only
The indicator argument will also accept string arguments, in which case the indicator function will use the value of the passed string as the name for the indicator column.

'''
 pd.merge(df1, df2, on='col1', how='outer', indicator='indicator_column')
'''
 
   col1 col_left  col_right indicator_column
0     0        a        NaN        left_only
1     1        b        2.0             both
2     2      NaN        2.0       right_only
3     2      NaN        2.0       right_only
Merge Dtypes
New in version 0.19.0.

Merging will preserve the dtype of the join keys.

'''
 left = pd.DataFrame({'k [1], ' [10]})

'''
 left
'''
 
   key  v1
0    1  10

'''
 right = pd.DataFrame({'k [1, 2], ' [20, 30]})

'''
 right
'''
 
   key  v1
0    1  20
1    2  30
We are able to preserve the join k

'''
 pd.merge(left, right, how='outer')
'''
 
   key  v1
0    1  10
1    1  20
2    2  30

'''
 pd.merge(left, right, how='outer').dtypes
'''
 
key    int64
v1     int64
dt object
Of course if you have missing values that are introduced, then the resulting dtype will be upcast.

'''
 pd.merge(left, right, how='outer', on='key')
'''
 
   key  v1_x  v1_y
0    1  10.0    20
1    2   NaN    30

'''
 pd.merge(left, right, how='outer', on='key').dtypes
'''
 
key       int64
v1_x    float64
v1_y      int64
dt object
New in version 0.20.0.

Merging will preserve category dtypes of the mergands. See also the section on categoricals.

The left frame.

'''
 from pandas.api.types import CategoricalDtype

'''
 X = pd.Series(np.random.choice(['foo', 'bar'], size=(10,)))

'''
 X = X.astype(CategoricalDtype(categories=['foo', 'bar']))

'''
 left = pd.DataFrame({ X,
   .                       np.random.choice(['one', 'two', 'three'],
   .                                            size=(10,))})
   . 

'''
 left
'''
 
     X      Y
0  bar    one
1  foo    one
2  foo  three
3  bar  three
4  foo    one
5  bar    one
6  bar  three
7  bar  three
8  bar  three
9  foo  three

'''
 left.dtypes
'''
 
X    category
Y      object
dt object
The right frame.

'''
 right = pd.DataFrame({ pd.Series(['foo', 'bar'],
   .                                      dtype=CategoricalDtype(['foo', 'bar'])),
   .                       [1, 2]})
   . 

'''
 right
'''
 
     X  Z
0  foo  1
1  bar  2

'''
 right.dtypes
'''
 
X    category
Z       int64
dt object
The merged res

'''
 result = pd.merge(left, right, how='outer')

'''
 result
'''
 
     X      Y  Z
0  bar    one  2
1  bar  three  2
2  bar    one  2
3  bar  three  2
4  bar  three  2
5  bar  three  2
6  foo    one  1
7  foo  three  1
8  foo    one  1
9  foo  three  1

'''
 result.dtypes
'''
 
X    category
Y      object
Z       int64
dt object
Note The category dtypes must be exactly the same, meaning the same categories and the ordered attribute. Otherwise the result will coerce to object dtype.
Note Merging on category dtypes that are the same can be quite performant compared to object dtype merging.
Joining on index
DataFrame.join() is a convenient method for combining the columns of two potentially differently-indexed DataFrames into a single result DataFrame. Here is a very basic exam

'''
 left = pd.DataFrame({ ['A0', 'A1', 'A2'],
   .                       ['B0', 'B1', 'B2']},
   .                     index=['K0', 'K1', 'K2'])
   . 

'''
 right = pd.DataFrame({ ['C0', 'C2', 'C3'],
   .                        ['D0', 'D2', 'D3']},
   .                      index=['K0', 'K2', 'K3'])
   . 

'''
 result = left.join(right)

#../_imag#../merging_join.png
'''
 result = left.join(right, how='outer')

#../_imag#../merging_join_outer.png
The same as above, but with how='inner'.

'''
 result = left.join(right, how='inner')

#../_imag#../merging_join_inner.png
The data alignment here is on the indexes (row labels). This same behavior can be achieved using merge plus additional arguments instructing it to use the inde

'''
 result = pd.merge(left, right, left_index=True, right_index=True, how='outer')

#../_imag#../merging_merge_index_outer.png
'''
 result = pd.merge(left, right, left_index=True, right_index=True, how='inner');

#../_imag#../merging_merge_index_inner.png
Joining key columns on an index
join() takes an optional on argument which may be a column or multiple column names, which specifies that the passed DataFrame is to be aligned on that column in the DataFrame. These two function calls are completely equival

left.join(right, on=key_or_keys)
pd.merge(left, right, left_on=key_or_keys, right_index=True,
      how='left', sort=False)
Obviously you can choose whichever form you find more convenient. For many-to-one joins (where one of the DataFrame’s is already indexed by the join key), using join may be more convenient. Here is a simple exam

'''
 left = pd.DataFrame({ ['A0', 'A1', 'A2', 'A3'],
   .                       ['B0', 'B1', 'B2', 'B3'],
   .                      'k ['K0', 'K1', 'K0', 'K1']})
   . 

'''
 right = pd.DataFrame({ ['C0', 'C1'],
   .                        ['D0', 'D1']},
   .                      index=['K0', 'K1'])
   . 

'''
 result = left.join(right, on='key')

#../_imag#../merging_join_key_columns.png
'''
 result = pd.merge(left, right, left_on='key', right_index=True,
   .                   how='left', sort=False);
   . 

#../_imag#../merging_merge_key_columns.png
To join on multiple keys, the passed DataFrame must have a MultiIn

'''
 left = pd.DataFrame({ ['A0', 'A1', 'A2', 'A3'],
   .                       ['B0', 'B1', 'B2', 'B3'],
   .                      'ke ['K0', 'K0', 'K1', 'K2'],
   .                      'ke ['K0', 'K1', 'K0', 'K1']})
   . 

'''
 index = pd.MultiIndex.from_tuples([('K0', 'K0'), ('K1', 'K0'),
   .                                   ('K2', 'K0'), ('K2', 'K1')])
   . 

'''
 right = pd.DataFrame({ ['C0', 'C1', 'C2', 'C3'],
   .                        ['D0', 'D1', 'D2', 'D3']},
   .                      index=index)
   . 
Now this can be joined by passing the two key column na

'''
 result = left.join(right, on=['key1', 'key2'])

#../_imag#../merging_join_multikeys.png
The default for DataFrame.join is to perform a left join (essentially a “VLOOKUP” operation, for Excel users), which uses only the keys found in the calling DataFrame. Other join types, for example inner join, can be just as easily perfor

'''
 result = left.join(right, on=['key1', 'key2'], how='inner')

#../_imag#../merging_join_multikeys_inner.png
As you can see, this drops any rows where there was no match.

Joining a single Index to a MultiIndex
You can join a singly-indexed DataFrame with a level of a MultiIndexed DataFrame. The level will match on the name of the index of the singly-indexed frame against a level name of the MultiIndexed frame.

'''
 left = pd.DataFrame({ ['A0', 'A1', 'A2'],
   .                       ['B0', 'B1', 'B2']},
   .                      index=pd.Index(['K0', 'K1', 'K2'], name='key'))
   . 

'''
 index = pd.MultiIndex.from_tuples([('K0', 'Y0'), ('K1', 'Y1'),
   .                                   ('K2', 'Y2'), ('K2', 'Y3')],
   .                                    names=['key', 'Y'])
   . 

'''
 right = pd.DataFrame({ ['C0', 'C1', 'C2', 'C3'],
   .                        ['D0', 'D1', 'D2', 'D3']},
   .                       index=index)
   . 

'''
 result = left.join(right, how='inner')

#../_imag#../merging_join_multiindex_inner.png
This is equivalent but less verbose and more memory efficien#../ faster than this.

'''
 result = pd.merge(left.reset_index(), right.reset_index(),
   .       on=['key'], how='inner').set_index(['key','Y'])
   . 

#../_imag#../merging_merge_multiindex_alternative.png
Joining with two MultiIndexes
This is supported in a limited way, provided that the index for the right argument is completely used in the join, and is a subset of the indices in the left argument, as in this exam

'''
 leftindex = pd.MultiIndex.from_product([list('abc'), list('xy'), [1, 2]],
   .                                        names=['abc', 'xy', 'num'])
   . 

'''
 left = pd.DataFrame({' range(12)}, index=leftindex)

'''
 left
'''
 
            v1
abc xy num    
a   x  1     0
       2     1
    y  1     2
       2     3
b   x  1     4
       2     5
    y  1     6
       2     7
c   x  1     8
       2     9
    y  1    10
       2    11

'''
 rightindex = pd.MultiIndex.from_product([list('abc'), list('xy')],
   ..                                         names=['abc', 'xy'])
   .. 

'''
 right = pd.DataFrame({' [100 * i for i in range(1, 7)]}, index=rightindex)

'''
 right
'''
 
         v2
abc xy     
a   x   100
    y   200
b   x   300
    y   400
c   x   500
    y   600

'''
 left.join(right, on=['abc', 'xy'], how='inner')
'''
 
            v1   v2
abc xy num         
a   x  1     0  100
       2     1  100
    y  1     2  200
       2     3  200
b   x  1     4  300
       2     5  300
    y  1     6  400
       2     7  400
c   x  1     8  500
       2     9  500
    y  1    10  600
       2    11  600
If that condition is not satisfied, a join with two multi-indexes can be done using the following code.

'''
 leftindex = pd.MultiIndex.from_tuples([('K0', 'X0'), ('K0', 'X1'),
   ..                                        ('K1', 'X2')],
   ..                                       names=['key', 'X'])
   .. 

'''
 left = pd.DataFrame({ ['A0', 'A1', 'A2'],
   ..                       ['B0', 'B1', 'B2']},
   ..                     index=leftindex)
   .. 

'''
 rightindex = pd.MultiIndex.from_tuples([('K0', 'Y0'), ('K1', 'Y1'),
   ..                                         ('K2', 'Y2'), ('K2', 'Y3')],
   ..                                        names=['key', 'Y'])
   .. 

'''
 right = pd.DataFrame({ ['C0', 'C1', 'C2', 'C3'],
   ..                        ['D0', 'D1', 'D2', 'D3']},
   ..                      index=rightindex)
   .. 

'''
 result = pd.merge(left.reset_index(), right.reset_index(),
   ..                   on=['key'], how='inner').set_index(['key', 'X', 'Y'])
   .. 

#../_imag#../merging_merge_two_multiindex.png
Merging on a combination of columns and index levels
New in version 0.23.

Strings passed as the on, left_on, and right_on parameters may refer to either column names or index level names. This enables merging DataFrame instances on a combination of index levels and columns without resetting indexes.

'''
 left_index = pd.Index(['K0', 'K0', 'K1', 'K2'], name='key1')

'''
 left = pd.DataFrame({ ['A0', 'A1', 'A2', 'A3'],
   ..                       ['B0', 'B1', 'B2', 'B3'],
   ..                      'ke ['K0', 'K1', 'K0', 'K1']},
   ..                     index=left_index)
   .. 

'''
 right_index = pd.Index(['K0', 'K1', 'K2', 'K2'], name='key1')

'''
 right = pd.DataFrame({ ['C0', 'C1', 'C2', 'C3'],
   ..                        ['D0', 'D1', 'D2', 'D3'],
   ..                       'ke ['K0', 'K0', 'K0', 'K1']},
   ..                      index=right_index)
   .. 

'''
 result = left.merge(right, on=['key1', 'key2'])

#../_imag#../merge_on_index_and_column.png
Note When DataFrames are merged on a string that matches an index level in both frames, the index level is preserved as an index level in the resulting DataFrame.
Note When DataFrames are merged using only some of the levels of a MultiIndex, the extra levels will be dropped from the resulting merge. In order to preserve those levels, use reset_index on those level names to move those levels to columns prior to doing the merge.
Note If a string matches both a column name and an index level name, then a warning is issued and the column takes precedence. This will result in an ambiguity error in a future version.
Overlapping value columns
The merge suffixes argument takes a tuple of list of strings to append to overlapping column names in the input DataFrames to disambiguate the result colu

'''
 left = pd.DataFrame({ ['K0', 'K1', 'K2'],  [1, 2, 3]})

'''
 right = pd.DataFrame({ ['K0', 'K0', 'K3'],  [4, 5, 6]})

'''
 result = pd.merge(left, right, on='k')

#../_imag#../merging_merge_overlapped.png
'''
 result = pd.merge(left, right, on='k', suffixes=['_l', '_r'])

#../_imag#../merging_merge_overlapped_suffix.png
DataFrame.join() has lsuffix and rsuffix arguments which behave similarly.

'''
 left = left.set_index('k')

'''
 right = right.set_index('k')

'''
 result = left.join(right, lsuffix='_l', rsuffix='_r')

#../_imag#../merging_merge_overlapped_multi_suffix.png
Joining multiple DataFrame or Panel objects
A list or tuple of DataFrames can also be passed to join() to join them together on their indexes.

'''
 right2 = pd.DataFrame({ [7, 8, 9]}, index=['K1', 'K1', 'K2'])

'''
 result = left.join([right, right2])

#../_imag#../merging_join_multi_df.png
Merging together values within Series or DataFrame columns
Another fairly common situation is to have two like-indexed (or similarly indexed) Series or DataFrame objects and wanting to “patch” values in one object from values for matching indices in the other. Here is an exam

'''
 df1 = pd.DataFrame([[np.nan, 3., 5.], [-4.6, np.nan, np.nan],
   ..                    [np.nan, 7., np.nan]])
   .. 

'''
 df2 = pd.DataFrame([[-42.6, np.nan, -8.2], [-5., 1.6, 4]],
   ..                    index=[1, 2])
   .. 
For this, use the combine_first() met

'''
 result = df1.combine_first(df2)

#../_imag#../merging_combine_first.png
Note that this method only takes values from the right DataFrame if they are missing in the left DataFrame. A related method, update(), alters non-NA values in pl

'''
 df1.update(df2)

#../_imag#../merging_update.png
Timeseries friendly merging
Merging Ordered Data
A merge_ordered() function allows combining time series and other ordered data. In particular it has an optional fill_method keyword to fi#../interpolate missing d

'''
 left = pd.DataFrame({ ['K0', 'K1', 'K1', 'K2'],
   ..                      ' [1, 2, 3, 4],
   ..                       ['a', 'b', 'c', 'd']})
   .. 

'''
 right = pd.DataFrame({ ['K1', 'K2', 'K4'],
   ..                       ' [1, 2, 3]})
   .. 

'''
 pd.merge_ordered(left, right, fill_method='ffill', left_by='s')
'''
 
     k   lv  s   rv
0   K0  1.0  a  NaN
1   K1  1.0  a  1.0
2   K2  1.0  a  2.0
3   K4  1.0  a  3.0
4   K1  2.0  b  1.0
5   K2  2.0  b  2.0
6   K4  2.0  b  3.0
7   K1  3.0  c  1.0
8   K2  3.0  c  2.0
9   K4  3.0  c  3.0
10  K1  NaN  d  1.0
11  K2  4.0  d  2.0
12  K4  4.0  d  3.0
Merging AsOf
New in version 0.19.0.

A merge_asof() is similar to an ordered left-join except that we match on nearest key rather than equal keys. For each row in the left DataFrame, we select the last row in the right DataFrame whose on key is less than the left’s key. Both DataFrames must be sorted by the key.

Optionally an asof merge can perform a group-wise merge. This matches the by key equally, in addition to the nearest match on the on key.

For example; we might have trades and quotes and we want to asof merge them.

'''
 trades = pd.DataFrame({
   ..     'ti pd.to_datetime(['2016052530:00.023',
   ..                             '2016052530:00.038',
   ..                             '2016052530:00.048',
   ..                             '2016052530:00.048',
   ..                             '2016052530:00.048']),
   ..     'tick ['MSFT', 'MSFT',
   ..                'GOOG', 'GOOG', 'AAPL'],
   ..     'pri [51.95, 51.95,
   ..               720.77, 720.92, 98.00],
   ..     'quanti [75, 155,
   ..                  100, 100, 100]},
   ..     columns=['time', 'ticker', 'price', 'quantity'])
   .. 

'''
 quotes = pd.DataFrame({
   ..     'ti pd.to_datetime(['2016052530:00.023',
   ..                             '2016052530:00.023',
   ..                             '2016052530:00.030',
   ..                             '2016052530:00.041',
   ..                             '2016052530:00.048',
   ..                             '2016052530:00.049',
   ..                             '2016052530:00.072',
   ..                             '2016052530:00.075']),
   ..     'tick ['GOOG', 'MSFT', 'MSFT',
   ..                'MSFT', 'GOOG', 'AAPL', 'GOOG',
   ..                'MSFT'],
   ..     'b [720.50, 51.95, 51.97, 51.99,
   ..             720.50, 97.99, 720.50, 52.01],
   ..     'a [720.93, 51.96, 51.98, 52.00,
   ..             720.93, 98.01, 720.88, 52.03]},
   ..     columns=['time', 'ticker', 'bid', 'ask'])
   .. 
'''
 trades
'''
 
                     time ticker   price  quantity
0 2016-05-2530:00.023   MSFT   51.95        75
1 2016-05-2530:00.038   MSFT   51.95       155
2 2016-05-2530:00.048   GOOG  720.77       100
3 2016-05-2530:00.048   GOOG  720.92       100
4 2016-05-2530:00.048   AAPL   98.00       100

'''
 quotes
'''
 
                     time ticker     bid     ask
0 2016-05-2530:00.023   GOOG  720.50  720.93
1 2016-05-2530:00.023   MSFT   51.95   51.96
2 2016-05-2530:00.030   MSFT   51.97   51.98
3 2016-05-2530:00.041   MSFT   51.99   52.00
4 2016-05-2530:00.048   GOOG  720.50  720.93
5 2016-05-2530:00.049   AAPL   97.99   98.01
6 2016-05-2530:00.072   GOOG  720.50  720.88
7 2016-05-2530:00.075   MSFT   52.01   52.03
By default we are taking the asof of the quotes.

'''
 pd.merge_asof(trades, quotes,
   ..               on='time',
   ..               by='ticker')
   .. 
'''
 
                     time ticker   price  quantity     bid     ask
0 2016-05-2530:00.023   MSFT   51.95        75   51.95   51.96
1 2016-05-2530:00.038   MSFT   51.95       155   51.97   51.98
2 2016-05-2530:00.048   GOOG  720.77       100  720.50  720.93
3 2016-05-2530:00.048   GOOG  720.92       100  720.50  720.93
4 2016-05-2530:00.048   AAPL   98.00       100     NaN     NaN
We only asof within 2ms between the quote time and the trade time.

'''
 pd.merge_asof(trades, quotes,
   ..               on='time',
   ..               by='ticker',
   ..               tolerance=pd.Timedelta('2ms'))
   .. 
'''
 
                     time ticker   price  quantity     bid     ask
0 2016-05-2530:00.023   MSFT   51.95        75   51.95   51.96
1 2016-05-2530:00.038   MSFT   51.95       155     NaN     NaN
2 2016-05-2530:00.048   GOOG  720.77       100  720.50  720.93
3 2016-05-2530:00.048   GOOG  720.92       100  720.50  720.93
4 2016-05-2530:00.048   AAPL   98.00       100     NaN     NaN
We only asof within 10ms between the quote time and the trade time and we exclude exact matches on time. Note that though we exclude the exact matches (of the quotes), prior quotes do propagate to that point in time.

'''
 pd.merge_asof(trades, quotes,
   ..               on='time',
   ..               by='ticker',
   ..               tolerance=pd.Timedelta('10ms'),
   ..               allow_exact_matches=False)
   .. 