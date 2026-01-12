#!/bin/bash
# Script to push branches and create PRs for quantum dots features
# Usage: ./create_prs.sh

set -e

REPO="qua-platform/qua-libs"
BASE_BRANCH="feat/quantum_dots"

# Colors for output
GREEN='\033[0;32m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

echo -e "${BLUE}=== Pushing branches to origin ===${NC}\n"

echo "Pushing feat/bayesian-change-point-detection..."
git push origin feat/bayesian-change-point-detection || echo "Branch may already exist on remote"

echo "Pushing feat/enhanced-charge-stability-analysis..."
git push origin feat/enhanced-charge-stability-analysis || echo "Branch may already exist on remote"

echo "Pushing feat/validation-utilities..."
git push origin feat/validation-utilities || echo "Branch may already exist on remote"

echo "Pushing feat/charge-stability-demo..."
git push origin feat/charge-stability-demo || echo "Branch may already exist on remote"

echo -e "\n${GREEN}All branches pushed!${NC}\n"

# Check if gh CLI is available
if command -v gh &> /dev/null; then
    echo -e "${BLUE}=== Creating PRs using GitHub CLI ===${NC}\n"
    
    # PR 1
    echo "Creating PR 1: Bayesian Change Point Detection..."
    gh pr create \
        --repo "$REPO" \
        --base "$BASE_BRANCH" \
        --head feat/bayesian-change-point-detection \
        --title "[Quantum Dots] Add Bayesian change point detection module" \
        --body "## Overview
This PR adds a comprehensive Bayesian change point detection module for quantum dot charge stability diagram analysis.

## Changes
- **New Module**: \`calibration_utils/bayesian_change_point/\`
  - \`bayesian_cp.py\`: Main Bayesian change point detection using MCMC
  - \`bayesian_lorentzian.py\`: Lorentzian mixture fitting with BIC model selection
  - \`bayesian_base.py\`: Base MCMC framework and fit result classes
  - \`standardization.py\`: Data standardization utilities
- **Integration**: Updates \`charge_stability/analysis.py\` to integrate Bayesian change point detection

## Features
- Bayesian MCMC-based change point detection for charge stability diagrams
- Lorentzian mixture fitting with automatic model selection via BIC
- Standardization utilities for data preprocessing
- Optional dependency handling for scikit-image

## Statistics
- **6 files changed**: +1,631 insertions
- **New files**: 5 module files + 1 integration file

## Target Branch
This PR targets \`feat/quantum_dots\` as part of the quantum dots calibration utilities enhancement."
    
    # PR 2
    echo "Creating PR 2: Enhanced Charge Stability Analysis..."
    gh pr create \
        --repo "$REPO" \
        --base "$BASE_BRANCH" \
        --head feat/enhanced-charge-stability-analysis \
        --title "[Quantum Dots] Enhance charge stability analysis with edge line detection" \
        --body "## Overview
This PR enhances the quantum dot charge stability analysis utilities with advanced edge line detection and improved plotting capabilities.

## Changes
- **New Module**: \`charge_stability/edge_line_analysis.py\` (376 lines)
  - Edge detection and skeletonization pipeline
  - Line segment fitting using total-least-squares
  - Intersection computation between segments
- **Enhanced Analysis**: \`charge_stability/analysis.py\`
  - Integration with edge line analysis
  - Improved change point detection workflows
- **Enhanced Plotting**: \`charge_stability/plotting.py\`
  - New plotting functions for change point overlays
  - Line fit overlays for charge state boundaries
  - Improved visualization capabilities
- **Updated Exports**: \`charge_stability/__init__.py\`
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
This PR targets \`feat/quantum_dots\` as part of the quantum dots calibration utilities enhancement."
    
    # PR 3
    echo "Creating PR 3: Validation Utilities..."
    gh pr create \
        --repo "$REPO" \
        --base "$BASE_BRANCH" \
        --head feat/validation-utilities \
        --title "[Quantum Dots] Add validation utilities for charge stability and time dynamics" \
        --body "## Overview
This PR adds comprehensive validation utilities for quantum dot simulations, including charge stability diagram simulation and time-dependent dynamics.

## Changes
- **Charge Stability Validation**: \`validation_utils/charge_stability/\`
  - \`default.py\`: Default six-dot charge-sensed quantum dot array model
  - Integration with \`qarray\` library for charge stability simulation
  - Configurable noise models (white noise, telegraph noise, latching model)
- **Time Dynamics Validation**: \`validation_utils/time_dynamics/\`
  - \`default.py\`: Time-dependent simulation utilities
  - \`src/device.py\`: Two-qubit device with configurable frequencies and coupling
  - \`src/circuit.py\`: Circuit class combining device + gates for time evolution
  - \`src/pulse.py\`: Pulse definitions for quantum gates
  - \`src/utils.py\`: Utility functions (embedding, sweeps, etc.)
- **Dependencies**: 
  - Added \`dynamiqs\` to dev dependencies in \`pyproject.toml\`
  - Updated \`poetry.lock\`

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
- \`qarray\` requires numpy > 2, which conflicts with current qm-qua version (documented in commit message)

## Target Branch
This PR targets \`feat/quantum_dots\` as part of the quantum dots calibration utilities enhancement."
    
    # PR 4
    echo "Creating PR 4: Demo Script..."
    gh pr create \
        --repo "$REPO" \
        --base "$BASE_BRANCH" \
        --head feat/charge-stability-demo \
        --title "[Quantum Dots] Add charge stability demo script and minor calibration updates" \
        --body "## Overview
This PR adds a comprehensive demo script for charge stability analysis and includes minor updates to existing calibration scripts.

## Changes
- **New Demo Script**: \`calibrations/loss_divincenzo/03b_charge_stability_demo.py\` (391 lines)
  - Complete workflow demonstration for 2D charge stability mapping
  - Integration with all analysis utilities (Bayesian, edge detection, plotting)
  - Support for simulation mode using validation utilities
  - Comprehensive documentation and examples
- **Minor Updates**:
  - \`00_close_other_qms.py\`: Minor import fix
  - \`03a_charge_stability.py\`: Minor update
  - \`charge_stability/parameters.py\`: Virtual sensor compensation integration

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
This PR targets \`feat/quantum_dots\` as part of the quantum dots calibration utilities enhancement."
    
    echo -e "\n${GREEN}All PRs created!${NC}"
else
    echo -e "${BLUE}GitHub CLI (gh) not found.${NC}"
    echo "Please create PRs manually using:"
    echo "  1. GitHub web interface: https://github.com/$REPO/compare"
    echo "  2. Or see PR_DESCRIPTIONS.md for detailed descriptions"
fi
