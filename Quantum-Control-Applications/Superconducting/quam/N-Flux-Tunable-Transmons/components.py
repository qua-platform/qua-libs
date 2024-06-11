"""This script facilitates the import of QuAM components from the parent folder `quam_components`, even in interactive mode.

Due to the QuAM components not being part of a standard Python module, scripts in this folder are unable to import them directly from the parent directory. 
This script addresses that limitation by adding the parent folder to the Python path, enabling the import of QuAM components.

Usage:
    Simply include this script at the beginning of your scripts in this folder, then run `from components import ...`

Note:
    Ensure this script is placed in the same directory as your other scripts that need to import QuAM components from the parent folder.
"""

from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).parent.parent))

from quam_components import *
