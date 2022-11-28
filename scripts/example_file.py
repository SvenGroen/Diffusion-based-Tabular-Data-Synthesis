import os, sys
# from tab_ddpm import GaussianMultinomialDiffusion
from subprocess import PIPE, run

def out(command):
    result = run(command, stdout=PIPE, stderr=PIPE, universal_newlines=True, shell=True)
    return result.stdout
print("\n\n")
print("-----INSIDE EXAMPLE_FILE.PY: -------")
print("Current wd:")
print(os.getcwd())
print("with files: ")
print(os.listdir(os.getcwd()))
print("Executable is: ", sys.executable)

print("ENV Variables found: ")
print(os.environ["PYTHONPATH"])
# os.system(f"export $PYTHONPATH={os.environ['PYTHONPATH']}")
print("Python and Conda version: ")
print(out("python --version"))
print(out("conda info --envs"))
# print("Conda packages: \n", os.system("conda list"))
# print("pip packages: \n", os.system("pip list"))


print("printing 'which python': ")
print(out("which python"))
print("echo pythonpath: ")
print(out("echo $PYTHONPATH"))
print("printing pythonpath with sys: ")
a=sys.path
for i in a:
    print(i)
print("Trying to load modules")
try:
    from tab_ddpm import GaussianMultinomialDiffusion
    print("LOADED GaussianMultinomialDiffusion SUCCESSFULLLY")
except Exception as e:
    print("COULD NOT LOAD TAB_DDPM, cause: ")
    print(e)
try:
    from lib.data import StandardScaler1d
    print("LOADED lib SUCCESSFULLLY")
except Exception as e:
    print("COULD NOT LOAD lib, cause: ")
    print(e)

print("---LEAVING EXAMPLE_FILE.PY: ----")
print("\n\n")