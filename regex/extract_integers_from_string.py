#from https://stackoverflow.com/questions/4289331/how-to-extract-numbers-from-a-string-in-python

import re

str = "h3110 23 cat 444.4 rabbit 11 2 dog"
[int(s) for s in str.split() if s.isdigit()]

#extract negative numbers
l = []
for t in s.split():
    try:
        l.append(float(t))
    except ValueError:
        pass

#counterexample using regex
re.findall(r'\b\d+\b', 'he33llo 42 I\'m a 32 string -30')
