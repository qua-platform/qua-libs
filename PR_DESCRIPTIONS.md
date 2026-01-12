# PR Descriptions for Quantum Dots Feature Branches

All PRs target: `feat/quantum_dots`

---

## PR 1: Bayesian Change Point Detection Module

**Branch**: `feat/bayesian-change-point-detection`  
**Target**: `feat/quantum_dots`  
**Title**: `[Quantum Dots] Add Bayesian change point detection module`

### Description:

```markdown
## Overview
This PR adds a comprehensive Bayesian change point detection module for quantum dot charge stability diagram analysis.

## Changes
- **New Module**: `calibration_utils/bayesian_change_point/`
  - `bayesian_cp.py`: Main Bayesian change point detection using MCMC
  - `bayesian_lorentzian.py`: Lorentzian mixture fitting with BIC model selection
  - `bayesian_base.py`: Base MCMC framework and fit result classes
  - `standardization.py`: Data standardization utilities
- **Integration**: Updates `charge_stability/analysis.py` to integrate Bayesian change point detection

## Features
- Bayesian MCMC-based change point detection for charge stability diagrams
- Lorentzian mixture fitting with automatic model selection via BIC
- Standardization utilities for data preprocessing
- Optional dependency handling for scikit-image

## Statistics
- **6 files changed**: +1,631 insertions
- **New files**: 5 module files + 1 integration file

## Target Branch
This PR targets `feat/quantum_dots` as part of the quantum dots calibration utilities enhancement.
```

---

## PR 2: Enhanced Charge Stability Analysis

**Branch**: `feat/enhanced-charge-stability-analysis`  
**Target**: `feat/quantum_dots`  
**Title**: `[Quantum Dots] Enhance charge stability analysis with edge line detection`

### Description:

```markdown
## Overview
This PR enhances the quantum dot charge stability analysis utilities with advanced edge line detection and improved plotting capabilities.

## Changes
- **New Module**: `charge_stability/edge_line_analysis.py` (376 lines)
  - Edge detection and skeletonization pipeline
  - Line segment fitting using total-least-squares
  - Intersection computation between segments
- **Enhanced Analysis**: `charge_stability/analysis.py`
  - Integration with edge line analysis
  - Improved change point detection workflows
- **Enhanced Plotting**: `charge_stability/plotting.py`
  - New plotting functions for change point overlays
  - Line fit overlays for charge state boundaries
  - Improved visualization capabilities
- **Updated Exports**: `charge_stability/__init__.py`
  - Added new analysis and plotting functions to public API

## Features
- Edge detection and skeletonization of charge stability diagrams
- Line segment fitting with orthogonal regression
- Automatic intersection detection between charge state boundaries
- Enhanced visualization with overlays and annotations

## Statistics
- **4 files changed**: +875 insertions, -11 deletions
- **New files**: 1 (edge_line_analysis.py)

## Dependencies
- Requires scikit-image for skeletonization
- May benefit from PR #1 (Bayesian change point detection) if merged first

## Target Branch
This PR targets `feat/quantum_dots` as part of the quantum dots calibration utilities enhancement.
```

---

## PR 3: Validation Utilities Module

**Branch**: `feat/validation-utilities`  
**Target**: `feat/quantum_dots`  
**Title**: `[Quantum Dots] Add validation utilities for charge stability and time dynamics`

### Description:

```markdown
## Overview
This PR adds comprehensive validation utilities for quantum dot simulations, including charge stability diagram simulation and time-dependent dynamics.

## Changes
- **Charge Stability Validation**: `validation_utils/charge_stability/`
  - `default.py`: Default six-dot charge-sensed quantum dot array model
  - Integration with `qarray` library for charge stability simulation
  - Configurable noise models (white noise, telegraph noise, latching model)
- **Time Dynamics Validation**: `validation_utils/time_dynamics/`
  - `default.py`: Time-dependent simulation utilities
  - `src/device.py`: Two-qubit device with configurable frequencies and coupling
  - `src/circuit.py`: Circuit class combining device + gates for time evolution
  - `src/pulse.py`: Pulse definitions for quantum gates
  - `src/utils.py`: Utility functions (embedding, sweeps, etc.)
- **Dependencies**: 
  - Added `dynamiqs` to dev dependencies in `pyproject.toml`
  - Updated `poetry.lock`

## Features
- Charge stability diagram simulation using qarray
- Time-dependent quantum dynamics using dynamiqs
- Two-qubit device simulation with configurable coupling
- Vectorized parameter sweeps using JAX
- Support for both lab frame and rotating frame simulations

## Statistics
- **14 files changed**: +3,097 insertions, -13 deletions
- **New files**: 10 validation utility files
- **Dependencies**: Added dynamiqs (^0.3.3) to dev group

## Note
- `qarray` requires numpy > 2, which conflicts with current qm-qua version (documented in commit message)

## Target Branch
This PR targets `feat/quantum_dots` as part of the quantum dots calibration utilities enhancement.
```

---

## PR 4: Demo Script + Minor Updates

**Branch**: `feat/charge-stability-demo`  
**Target**: `feat/quantum_dots`  
**Title**: `[Quantum Dots] Add charge stability demo script and minor calibration updates`

### Description:

```markdown
## Overview
This PR adds a comprehensive demo script for charge stability analysis and includes minor updates to existing calibration scripts.

## Changes
- **New Demo Script**: `calibrations/loss_divincenzo/03b_charge_stability_demo.py` (391 lines)
  - Complete workflow demonstration for 2D charge stability mapping
  - Integration with all analysis utilities (Bayesian, edge detection, plotting)
  - Support for simulation mode using validation utilities
  - Comprehensive documentation and examples
- **Minor Updates**:
  - `00_close_other_qms.py`: Minor import fix
  - `03a_charge_stability.py`: Minor update
  - `charge_stability/parameters.py`: Virtual sensor compensation integration

## Features
- Complete charge stability analysis workflow demonstration
- Integration of Bayesian change point detection
- Edge line analysis and visualization
- Simulation support for validation
- Virtual sensor compensation for plunger gates

## Statistics
- **4 files changed**: +395 insertions, -3 deletions
- **New files**: 1 (03b_charge_stability_demo.py)

## Dependencies
This PR demonstrates the integration of features from:
- PR #1: Bayesian change point detection
- PR #2: Enhanced charge stability analysis
- PR #3: Validation utilities

**Recommended merge order**: After PRs #1, #2, and #3 (or at least #1 and #2)

## Target Branch
This PR targets `feat/quantum_dots` as part of the quantum dots calibration utilities enhancement.
```

---

## Instructions to Create PRs

### Option 1: Using GitHub CLI (gh)

```bash
# Push all branches first
git push origin feat/bayesian-change-point-detection
git push origin feat/enhanced-charge-stability-analysis
git push origin feat/validation-utilities
git push origin feat/charge-stability-demo

# Create PR 1
gh pr create --base feat/quantum_dots --head feat/bayesian-change-point-detection \
  --title "[Quantum Dots] Add Bayesian change point detection module" \
  --body-file <(cat PR_DESCRIPTIONS.md | sed -n '/## PR 1:/,/^---$/p' | sed -n '/^```markdown$/,/^```$/p' | sed '1d;$d')

# Create PR 2
gh pr create --base feat/quantum_dots --head feat/enhanced-charge-stability-analysis \
  --title "[Quantum Dots] Enhance charge stability analysis with edge line detection" \
  --body-file <(cat PR_DESCRIPTIONS.md | sed -n '/## PR 2:/,/^---$/p' | sed -n '/^```markdown$/,/^```$/p' | sed '1d;$d')

# Create PR 3
gh pr create --base feat/quantum_dots --head feat/validation-utilities \
  --title "[Quantum Dots] Add validation utilities for charge stability and time dynamics" \
  --body-file <(cat PR_DESCRIPTIONS.md | sed -n '/## PR 3:/,/^---$/p' | sed -n '/^```markdown$/,/^```$/p' | sed '1d;$d')

# Create PR 4
gh pr create --base feat/quantum_dots --head feat/charge-stability-demo \
  --title "[Quantum Dots] Add charge stability demo script and minor calibration updates" \
  --body-file <(cat PR_DESCRIPTIONS.md | sed -n '/## PR 4:/,/^---$/p' | sed -n '/^```markdown$/,/^```$/p' | sed '1d;$d')
```

### Option 2: Using GitHub Web Interface

1. Push all branches:
   ```bash
   git push origin feat/bayesian-change-point-detection
   git push origin feat/enhanced-charge-stability-analysis
   git push origin feat/validation-utilities
   git push origin feat/charge-stability-demo
   ```

2. Go to https://github.com/qua-platform/qua-libs/compare
3. For each branch:
   - Select base: `feat/quantum_dots`
   - Select compare: the feature branch
   - Click "Create pull request"
   - Copy the title and description from this file
   - Submit

### Option 3: Using GitKraken

1. Push all branches (as above)
2. Sign in to GitKraken if needed
3. Use the GitKraken UI to create PRs with the descriptions above
