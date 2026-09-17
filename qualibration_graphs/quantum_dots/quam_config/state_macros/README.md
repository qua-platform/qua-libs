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

1. a macro class inheriting from `CustomMacro`
2. configurable fields declared directly on that macro dataclass
3. an `apply(...)` method containing the QUA logic
4. an `inferred_duration` property describing the macro duration, when known

Example:

```python
@quam_dataclass
class MyInitializeMacro(CustomMacro):
    ramp_duration: int = 200
    hold_duration: int = 400

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

Every `CustomMacro` subclass automatically exposes a `Parameters` attribute. 
The `Parameters` attribute is the bridge between a macro's necessary attributes, 
and a Qualibrate node's `Parameters` usage. `CustomMacro` auto-generates the 
`Parameters` class by reading the macro's class attributes. 

`my_macros.py` uses it like this:

```python
class MacroParameters(
    RunnableParameters,
    initialize_macro.Parameters,
    measure_macro.Parameters,
):
    pass
```

This means that any fields declared on the macro class can be exposed in a node's 
`parameters.py` files without manually creating a separate parameters class. 

## What `resolve_params(...)` Does

`resolve_params(**kwargs)` is a convenience helper provided by `CustomMacro`.
It builds a dictionary by iterating over the macro's dataclass fields and, for
each field name, choosing between:

- the value currently stored on the macro instance, `self.<field_name>`
- the override passed in `kwargs[field_name]`, if that override is not `None`

More precisely, for every dataclass field on the macro, it does the equivalent
of:

```python
override = kwargs.get(field_name)
value = self.<field_name> if override is None else override
```

Two details are important:

- it only considers dataclass field names already defined on the macro
- passing `None` is treated the same as not passing an override at all

So `resolve_params(...)` is convenient to be able to pass override arguements
into the macro's `apply(...)` without manually listing them all in the function
declaration. 

**Importantly, this is optional**. You **do not need** to use `resolve_params(...)` 
in a custom macro. You can just as easily declare explicit arguments on `apply(...)`
and perform the fallback logic yourself. Both approaches work; `resolve_params(...)`
is just a convenience for macros that have several overridable fields, allowing you to 
centralize the attributes logic. 

### Pattern 1: using `resolve_params(...)`

```python
params = self.resolve_params(**kwargs)
ramp = params["ramp_duration"]
hold = params["hold_duration"]
```

Example:

```python
@quam_dataclass
class MyInitializeMacro(CustomMacro):
    ramp_duration: int = 200
    hold_duration: int = 400

    @property
    def inferred_duration(self) -> float | None:
        return (self.ramp_duration + self.hold_duration) * 1e-9

    def apply(self, **kwargs):
        params = self.resolve_params(**kwargs)
        ramp = params["ramp_duration"]
        hold = params["hold_duration"]
        ...
```

Calling:

```python
my_macro.apply(ramp_duration=800)
```

will use:

- `ramp_duration = 800`
- `hold_duration = self.hold_duration`

### Pattern 2: explicit arguments without `resolve_params(...)`

The same behavior can be written out directly:

```python
@quam_dataclass
class MyInitializeMacro(CustomMacro):
    ramp_duration: int = 200
    hold_duration: int = 400

    @property
    def inferred_duration(self) -> float | None:
        return (self.ramp_duration + self.hold_duration) * 1e-9

    def apply(
        self,
        ramp_duration: int | None = None,
        hold_duration: int | None = None,
        **kwargs,
    ):
        ramp = self.ramp_duration if ramp_duration is None else ramp_duration
        hold = self.hold_duration if hold_duration is None else hold_duration
        ...
```

This can be easier to read when a macro has only a few arguments, or when you
want the `apply(...)` signature itself to document the expected keyword names.

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
