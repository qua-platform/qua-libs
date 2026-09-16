# Custom State Macros

This folder contains custom initialize and measure macros for the quantum-dot
QUAM.

Reference implementation details for the underlying macro infrastructure can
be found in the
[`quam-builder` operations package](https://github.com/qua-platform/quam-builder/tree/main/quam_builder/architecture/quantum_dots/operations).

If you are editing macros for the first time, the goal is simple:

1. define the macro logic here
2. select the macro in `../my_macros.py`
3. run `python ../my_macros.py` to wire it into the machine state

## Recommended Pattern

Each macro in this folder should usually have:

1. an attributes class containing the configurable fields
2. a macro class inheriting from both `CustomMacro` and that attributes class
3. a `Parameters` class attribute pointing at the attributes class
4. an `apply(...)` method containing the QUA logic
5. an `inferred_duration` property describing the macro duration, when known

Example:

```python
@quam_dataclass
class MyInitializeAttributes:
    ramp_duration: int = 200
    hold_duration: int = 400


@quam_dataclass
class MyInitializeMacro(CustomMacro, MyInitializeAttributes):
    Parameters = MyInitializeAttributes

    @property
    def inferred_duration(self) -> float | None:
        return (self.ramp_duration + self.hold_duration) * 1e-9

    def apply(self, **kwargs):
        params = self.resolve_params(**kwargs)
        ramp = params["ramp_duration"]
        hold = params["hold_duration"]
        ...
```

## What `Parameters` Means

The `Parameters` class attribute is the bridge between a macro and the IDE
parameters shown in Qualibrate nodes.

`my_macros.py` uses it like this:

```python
class MacroParameters(
    RunnableParameters,
    initialize_macro.Parameters,
    measure_macro.Parameters,
):
    pass
```

That means any fields placed on the macro's attributes class can be exposed in
node `parameters.py` files without separately importing a second "parameter"
class by hand.

## What `resolve_params(...)` Does

`resolve_params(**kwargs)` merges two sources of values:

- the values stored on the macro instance itself
- any explicit keyword overrides passed in the current call

This lets a macro have persistent defaults while still allowing one-off
per-call overrides from a node or helper script.

In practice, a pattern like:

```python
params = self.resolve_params(**kwargs)
ramp = params["ramp_duration"]
```

means:

- use `kwargs["ramp_duration"]` if it was passed in this call
- otherwise fall back to `self.ramp_duration`

## What `inferred_duration` Is For

`inferred_duration` is the macro's best estimate of how long it takes to run,
in seconds.

It is useful when:

- other code wants to reason about timing
- a larger macro is built from smaller macros
- the duration can be derived from the macro fields

If the duration cannot be known ahead of time, returning `None` is fine.

## Who `owner` Usually Is

For these state macros, the `owner` is designed to be a `QuantumDotPair`. The reason for this 
is that in many spin qubit/quantum dot experiments, initialization and measurement is done 
pair-wise; our Quam structure mirrors this real, physical experimental structure. 

That is why many macros access things like:

- `owner.voltage_sequence`
- `owner.measure(...)`
- `owner.machine`

Because the macro is wired at the `QuantumDotPair` level, a single pair-level
macro is reused underneath calls such as:

- `qubit.initialize()`
- `qubit_pair.initialize()`
- `dot_pair.initialize()`

The same idea applies to measure macros.

## Where To Start

- `initialize_macros.py`: edit this when you want to customize how the system
  is prepared before an experiment.
- `measure_macros.py`: edit this when you want to customize the readout flow or
  expose readout-specific parameters.
