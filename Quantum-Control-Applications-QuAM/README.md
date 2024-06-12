# Quantum Control Applications

This folder contains scripts and examples for different qubits types, all the way from basic to advanced protocols. 
It also includes various examples and results from labs as listed below.

Note that for them to work, you would need to download the latest version of the [py-qua-tools](https://github.com/qua-platform/py-qua-tools#installation), as well as [QuAM](https://github.com/qua-platform/quam) and [quam_components](https://github.com/qua-platform/quam_components).

## Superconducting Qubits
### [N-flux tunable transmons](https://github.com/qua-platform/qua-libs/tree/main/Quantum-Control-Applications/Superconducting/Two-Flux-Tunable-Transmons)
These files showcase various experiments that can be done on a several flux-tunable transmons.

# Convert the files to ipynb
If you wish to work with Jupyter notebooks, then you will need to convert the .py files into .ipynb files. 
To do so you can either do it manually or install the `ipynb-py-convert` package that can then be called from Jupyter 
notebook cell or python file as in:
```python
import os
import ipynb_py_convert
# iterate over files in current directory
for filename in os.listdir():
    f = os.path.join(filename)
    # checking if it is a file
    if os.path.isfile(f):
        # checking if it is a python file that we want to convert
        if f[-3:] == ".py" and (f[0] in ["0", "1", "2"]):
            print(f[:-2]+"ipynb")
            ipynb_py_convert.convert(f, f[:-2]+"ipynb")
```

Moreover, in order to be able to change the parameters in the configuration.py file and update the jupyter kernel, you will need 
to replace in all the files the line ``from configuration import *`` by 
``` python
from importlib import reload
import configuration
reload(configuration)
from configuration import *
```
This can be done easily by typing `Ctrl + shift + F` in VS Code or `Ctrl + shift + R` in PyCharm.

Finally, if you wish to benefit from the live plotting feature, then we advise you to work with the qt backend that can 
be enabled in a given notebook by adding the magic line `%matplotlib qt`.
To do it in all files, you can replace the line `import matplotlib.pyplot as plt` by
``` python
import matplotlib.pyplot as plt
%matplotlib qt
```
using the shortcut command above. Note that you may need to install the PyQt package if it is not already in your python environment.