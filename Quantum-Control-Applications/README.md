# Quantum Control Applications

This folder contains scripts and examples for different qubits types, all the way from basic to advanced protocols. 
It also includes various examples and results from labs as listed below.

Note that for them to work, you would need to download the latest version of the [py-qua-tools](https://github.com/qua-platform/py-qua-tools#installation).

## Superconducting Qubits
### [Two flux tunable transmons](https://github.com/qua-platform/qua-libs/tree/main/Quantum-Control-Applications/Superconducting/Two-Flux-Tunable-Transmons)
These files showcase various experiments that can be done on a several flux-tunable transmons using the [standard configuration](https://github.com/qua-platform/qua-libs/tree/main/Quantum-Control-Applications/Superconducting/Two-Flux-Tunable-Transmons/Standard%20Configuration#two-flux-tunable-transmons-with-the-standard-configuration).
#### <u> Advanced use-cases: </u>
* [SWAP spectroscopy improved with predistortion digital filters](https://github.com/qua-platform/qua-libs/tree/main/Quantum-Control-Applications/Superconducting/Two-Flux-Tunable-Transmons/Use%20Case%201%20-%20Two%20qubit%20gate%20optimization%20with%20cryoscope#two-qubit-swap-spectroscopy-improved-with-pre-distortion-digital-filters).
* [Two-qubit randomized benchmarking](https://github.com/qua-platform/qua-libs/tree/main/Quantum-Control-Applications/Superconducting/Two-Flux-Tunable-Transmons/Use%20Case%202%20-%20Two-Qubit-Randomized-Benchmarking#two-qubit-randomized-benchmarking).
* [Two-qubit cross-entropy benchmarking](https://github.com/qua-platform/qua-libs/tree/main/Quantum-Control-Applications/Superconducting/Two-Flux-Tunable-Transmons/Use%20Case%203%20-%20Two-Qubit%20Cross-Entropy%20Benchmarking).

### [Single fixed frequency transmon](https://github.com/qua-platform/qua-libs/tree/main/Quantum-Control-Applications/Superconducting/Single-Fixed-Transmon#single-fixed-transmon-superconducting-qubit)
These files showcase various experiments that can be done on a single fixed-frequency transmon.
#### <u> Advanced use-cases: </u>
* [Qubit Frequency Tracking](https://github.com/qua-platform/qua-libs/tree/main/Quantum-Control-Applications/Superconducting/Single-Fixed-Transmon/Use%20Case%201%20-%20Schuster%20Lab%20-%20Qubit%20Frequency%20Tracking#qubit-frequency-tracking) 
  performed in the lab of Prof. David Schuster in the University of Chicago.
* [Optimized readout with optimal weights](https://github.com/qua-platform/qua-libs/tree/main/Quantum-Control-Applications/Superconducting/Single-Fixed-Transmon/Use%20Case%202%20-%20Optimized%20readout%20with%20optimal%20weights#optimized-readout-with-optimal-weights).

### [Single flux tunable transmon](https://github.com/qua-platform/qua-libs/tree/main/Quantum-Control-Applications/Superconducting/Single-Flux-Tunable-Transmon#single-flux-tunable-transmon)
These files showcase various experiments that can be done on a single flux-tunable transmon.
#### <u> Advanced use-cases: </u>
* [Parametric Drive between flux-tunable-qubit and qubit-coupler](https://github.com/qua-platform/qua-libs/tree/main/Quantum-Control-Applications/Superconducting/Single-Flux-Tunable-Transmon/Use%20Case%203%20-%20Ma%20Lab%20-%20Parametric%20Drive%20iSWAP#parametric-drive-between-flux-tunable-qubit-and-qubit-coupler) 
  performed in the lab of Prof. Alex Ma at Purdue University.
* [Cryoscope](https://github.com/qua-platform/qua-libs/tree/main/Quantum-Control-Applications/Superconducting/Single-Flux-Tunable-Transmon/Use%20Case%201%20-%20Paraoanu%20Lab%20-%20Cryoscope#cryoscope) 
  performed in the lab of Prof. Sorin Paraoanu in Aalto University.
* [DRAG Pulse Calibration](https://github.com/qua-platform/qua-libs/tree/main/Quantum-Control-Applications/Superconducting/Single-Flux-Tunable-Transmon/Use%20Case%202%20-%20DRAG%20coefficient%20calibration#derivative-removal-by-adiabatic-gate-drag-and-ac-stark-shift-calibration).

## [AMO Qubits](https://github.com/qua-platform/qua-libs/tree/main/Quantum-Control-Applications/AMO)
### Rydberg Arrays

#### <u> Advanced use-cases: </u>
* [2D Atom Sorting](https://github.com/qua-platform/qua-libs/tree/main/Quantum-Control-Applications/AMO/Use%20Case%201%20-%20Saffman%20Lab%20-%20Atom%20Sorting#atom-sorting-with-the-opx)
  performed in the lab of Prof. Mark Saffman in the University of Wisconsin-Madison.

## Optically addressable spin qubits

### [Cryogenic nanophotonic cavity](https://github.com/qua-platform/qua-libs/tree/main/Quantum-Control-Applications/Optically%20addressable%20spin%20qubits/Cryogenic%20nanophotonic%20cavity#single-yb-center-in-a-cyrogenic-nanophotonic-cavity)
These files showcase various experiments that can be done on a Yb center (also works for other rare-earth ions) in a 
cryogenic nanophotonic cavity with a SNSPD and an AOM which is controlled via a digital channel.
#### <u> Advanced use-cases: </u>
* [High resolution time-tagging](https://github.com/qua-platform/qua-libs/tree/main/Quantum-Control-Applications/Optically%20addressable%20spin%20qubits/Cryogenic%20nanophotonic%20cavity/Use%20case%201%20-%20Faraon%20Lab%20-%20sub-ns%20timetagging#high-resolution-time-tagging)
  performed in the lab of Prof. Faraon at Caltech.
### [Electron Spin Resonance set-up](https://github.com/qua-platform/qua-libs/tree/main/Quantum-Control-Applications/Optically%20addressable%20spin%20qubits/Electron%20Spin%20Resonance#electron-spin-resonance-esr-experiments)
These ESR protocols can be used in a variety of ensemble of defects in solids such as NV.
#### <u> Advanced use-cases: </u>
* [CPMG](https://github.com/qua-platform/qua-libs/tree/main/Quantum-Control-Applications/Optically%20addressable%20spin%20qubits/Electron%20Spin%20Resonance/Use%20case%201%20-%20Sekhar%20Lab%20-%20CPMG#carr-purcell-meiboom-gill-cpmg-in-an-nv-ensemble-with-electron-spin-resonance-esr)
  performed in the lab of Prof. Sekhar at Darmouth College.

### [NV center in a confocal setup](https://github.com/qua-platform/qua-libs/tree/main/Quantum-Control-Applications/Optically%20addressable%20spin%20qubits/NV%20center%20in%20a%20confocal%20setup#single-nv-center-in-a-confocal-setup)
These files showcase various experiments that can be done on an NV center in a confocal setup with an SPCM and an AOM
which is controlled via a digital channel.

## Quantum-dots
### [Single Spin with EDSR](https://github.com/qua-platform/qua-libs/blob/main/Quantum-Control-Applications/Quantum-Dots/Single_Spin_EDSR/README.md)
These files showcase various experiments that can be done on a single spin driven by Electric Dipole Spin Resonance (EDSR).
Set-ups including the Octave and/or QDAC2 are also supported.

### [Singlet-Triplet qubit](https://github.com/qua-platform/qua-libs/blob/main/Quantum-Control-Applications/Quantum-Dots/Singlet_Triplet_Qubit/README.md)
These files showcase various experiments that can be done on a singlet-triplet qubit.
Set-ups including the QDAC2 are also supported.

#### <u> Advanced use-cases: </u>
* [Fast 2D scans using a spiral pattern](https://github.com/qua-platform/qua-libs/tree/main/Quantum-Control-Applications/Quantum-Dots/Use%20Case%201%20-%20Fast%202D%20Scans#fast-two-dimensional-scans-using-a-spiral-pattern)
  performed in the lab of Prof. Natalia Ares in the University of Oxford.

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