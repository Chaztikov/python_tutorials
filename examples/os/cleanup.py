import os


def take_out_trash(cwd="$HOME/", extensions="*.o"):
    try:
        command = "rm -rf " + cwd + "/" + extensions
        os.system("git " + command)
        os.system(command)
    except Exception as e:
        print(e)


def directory_purge(cwd="$HOME/", extensions="*.o"):
    directories = [cwd + "/" + directory for directory in os.listdir(
        cwd) if os.path.isdir(cwd + "/" + directory)]
    for directory in directories:
        try:
            directory_purge(directory, extensions)
        except Exception as e:
            print(e)
            take_out_trash(directory, extensions)


def destroy(cwd="$HOME/", extensions="*.o"):
    directories = [cwd + "/" + directory for directory in os.listdir(
        cwd) if not os.path.isfile(directory)]
    for directory in directories:
        command = "rm -rf " + extensions
        try:
            os.system("git " + command)
        except Exception as e:
            print(e)
        try:
            os.system(command)
        except Exception as e:
            print(e)

    import os
    from os.path import join, getsize
    for root, dirs, files in os.walk(os.getcwd()):
        print(root, "consumes", end="")
        print(sum([getsize(join(root, name)) for name in files]), end="")
        print("bytes in", len(files), "non-directory files")
        if '.vscode' in dirs:
            dirs.remove('CVS')  # don't visit CVS directories
        for file in files:
            if file.endswith('-opt'):
                file.delete()
                
cwd = os.getcwd()
extensions = "*.o *.e example-opt example-dbg .ipynb*"
directory_purge(cwd, extensions)
