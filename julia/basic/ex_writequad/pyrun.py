import numpy as np;
import os,sys

wdir="/home/chaztikov/git/nonlocal-discretization-tools/minitest/basic/ex_trap"

print(wdir)

os.chdir(wdir)

os.system("make -j12 && rm -f *.e && ./example-opt")



files = [file for file in os.listdir() if file.endswith(".e")]

data=[]
for file in files:
	dat = np.loadtxt(file)
	print("\n ",file, "\n ", dat.shape)
	data.append(dat.tolist())

data = np.array(data)

