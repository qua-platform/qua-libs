# QUA Coding Rules (Authoritative)

- All QUA code must be inside `with program() as prog:`
- Python `for`, `while`, `if` are NOT allowed; use `for_`, `while_`, `if_`
- QUA variables must be declared using `declare(type)`
- Allowed QUA types: int, fixed, bool, stream
- Do not use Python lists or dicts inside QUA programs
- Timing is hardware-defined; avoid Python-side delays
- All measurements must use `measure()` with a defined element
