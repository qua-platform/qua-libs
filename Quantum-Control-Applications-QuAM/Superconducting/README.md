# N Flux-Tunable Transmon Qubits
## Installation
This folder contains an installable module called `quam_components`, which provides a collection of tailored components for controlling flux-tunable qubits. These components extend the functionality of QuAM, making it easier to design and execute calibration nodes.

### Requirements
To run the calibration nodes in this folder, you need to install the `quam_components`. First, ensure you have Python ≥ 3.8 installed on your system.
Then run the following command:

```sh
# Install quam
pip install git+https://github.com/qua-platform/quam.git
# Install quam_components
pip install -e . 
```
> **_NOTE:_**  The `-e` flag means you *don't* have to reinstall if you make a local change to `quam_components`!