module.exports = [
    'examples_index',
    {
        "type": "category",
        "label": "Basics",
        "items": [
            "basics/hello-qua/index",
            "basics/basic-digital-output/index",
            "basics/intro-to-saving/index",
            "basics/intro-to-streams/index",
            "basics/raw-adc-measurement/index",
            "basics/frame-and-phase-intro/index",
            "basics/waveform-compression/index",
            "basics/intro-to-macros/index",
            "basics/intro-to-integration/index",
            "basics/intro-to-demod/index",
        ]
    },
    {
        "type": "category",
        "label": "advanced-topics",
        "items": [
            "advanced-topics/single-sideband-modulation/index",
            "filters/index"
        ]
    },
    {
        "type": "category",
        "label": "Characterization",
        "items": [
            "characterization/active-reset/index",
            "characterization/qubit-state-tomography/index",
            "characterization/qubit-process-tomography/index",
            "characterization/wigner-tomography/index",
        ]
    },
    {
        "type": "category",
        "label": "Calibrations",
        "items": [
        {
          type: 'category',
          label: 'T1',
          items: ["calibration/T1/superconducting-qubits/index"]
          },

        {
          type: 'category',
          label: 'T2',
          items: ["calibration/T2/superconducting-qubits/index"]
          },
//            "calibration/T2/index",
            "calibration/rabi-sweeps/index",
            "calibration/rabi-sweeps/helper-for-high-res-time-rabi/index",
            "calibration/hahn-echo/index",
        ]
    },
    {
        "type": "category",
        "label": "Dynamical Decoupling Protocols",
        "items": [
            "dynamical-decoupling-protocols/XY-n/index",
            "dynamical-decoupling-protocols/CPMG/index"
        ]
    },
    {
        "type": "category",
        "label": "Multi level and multiplexed readout",
        "items": [
//            "multi-qubit/flux-tuneable-coupler/index",
            "multi-qubit/multilevel-discriminator/index",
            "multi-qubit/multiplexed-multilevel-NN-discriminator/index",
            "multi-qubit/multiplexed-readout/index"
        ]
    },
    {
        "type": "category",
        "label": "Advanced algorithms",
        "items": [
            "multi-qubit/VQA/QAOA/index",
            "multi-qubit/QRAM/index",

        ]
    },
    {
        "type": "category",
        "label": "Randomized Benchmark",
        "items": [
            "randomized-benchmark/one-qubit-rb/index",
            "randomized-benchmark/DRAG-optimization/index"
        ]
    },
    {
        "type": "category",
        "label": "Spectroscopy",
        "items": [
            "spectroscopy/qubit-spectroscopy/index",
            "spectroscopy/resonator-spectroscopy/index",
        ]
    },
    {
        "type": "category",
        "label": "Tomography",
        "items": [
            "characterization/qubit-state-tomography/index",
            "characterization/wigner-tomography/index",
        ]
    },
    {
        "type": "category",
        "label": "NV Centers",
        "items": [
            "nv-centers/syncing-opx-with-external-devices/index",
//            "nv-centers/g2-with-stage/index",  Need to finished readme
//            "nv-centers/widefield-odmr/index",  Need to finished readme
        ]
    },
]