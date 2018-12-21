import os
from os.path import join, getsize

extensions = "*.o *.e example-opt example-dbg .ipynb*"
extensions = extensions.split()
print(extensions)
for root, dirs, files in os.walk(os.getcwd()):
    print(root, "consumes", end="")
    print(sum([getsize(join(root, name)) for name in files]), end="")
    print("bytes in", len(files), "non-directory files")
    if '.vscode' in dirs:
        dirs.remove('CVS')  # don't visit CVS directories
    for di in dirs:
        for file in files:
            for ext in extensions:
                if file.endswith(ext):
                    fpath=root +"/"+ di +"/"+ file
                    print(fpath)
                    try:
                        os.remove(fpath)
                        pass
                    except Exception as e:
                        print(e)
