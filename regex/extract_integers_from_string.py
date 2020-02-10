#from https://stackoverflow.com/questions/4289331/how-to-extract-numbers-from-a-string-in-python

str = "h3110 23 cat 444.4 rabbit 11 2 dog"
[int(s) for s in str.split() if s.isdigit()]
