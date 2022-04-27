---
title: Introduction to saving
sidebar_label: Intro to saving
slug: ./
id: index
---

This example shows five QUA programs where variables are saved.
Each one shows a slightly different variation on how this can be achieved:
either by saving directly to a tag that is then collected from the `result_handles`
structure, or by saving to a `stream` and processing it in various ways. 

## Config

The configuration for this example is included, but irrelevant as no pulses are 
played to any output and no data is read in. 

## Programs

Four programs are included: 
1. Assigning values to variables and saving variables to tags.
This program saves literal values and values calculated with math operations. 
It then saves them directly to `tags`. This is a less powerful method of saving
that is not the recommended mode of operation, yet it is still supported as a legacy method. 
```python
with program() as saving_a_var:
    a = declare(int, value=5)
    b = declare(fixed, value=0.23)
    save(a, "a_var")
    assign(a, 7)
    save(a, "a_var")
    assign(a, a + 1)
    save(a, "a_var")
    assign(b, math.sin2pi(b))
    save(b, "b_var")
```
2. Saving variables to streams and using stream processing.
The `stream` construct is a powerful way to save and manipulate data on the server. 
It is described in detail in the [QUA docs](https://qm-docs.qualang.io/guides/stream_proc). This example shows basic usage. 
```python
with program() as streamProg:
    out_str = declare_stream()
    a = declare(int)
    with for_(a, 0, a <= 100, a + 1):
        save(a, out_str)

    with stream_processing():
        # Average all of the data and save only the last value into "out".
        out_str.average().save("out")
```
3. Using the buffer operator in stream processing.

Streams can be acted upon with various operators. The buffer operator is a 
particularly useful one which allows to reshape incoming data into buffers of 
a predefined size. Using additional operators, such as the `average()` operator 
as shown here, allows averaging such buffers with one another. In this case 
all average buffers are saved, such that you can track the evolution of the average.  
```python
with program() as streamProg_buffer:
    out_str = declare_stream()

    a = declare(int)
    with for_(a, 0, a <= 100, a + 1):
        save(a, out_str)

    with stream_processing():
        # Group output into vectors of length 3. Since only full buffers are used,
        # the last 2 data points [99 100] are discarded.
        # Perform a running average over the data, in group of 3:
        # The first vector is [0 1 2] and it averages only with itself.
        # The second vector is [3 4 5] and it averages with the 1st vector, giving [1.5 2.5 3.5].
        # etc...
        # This time 'save_all' is used, so all of the data is saved ('save' would have only saved [48 49 50])
        out_str.buffer(3).average().save_all("out")
```
4. Saving a stream to multiple tags.

This examples shows how multiple output variables can be defined in one `steam_processing`
environment. 
```python
with program() as multiple_tags:
    out_str1 = declare_stream()

    a = declare(int)
    with for_(a, 0, a <= 100, a + 1):
        save(a, out_str1)

    with stream_processing():
        # Two separate streams (or pipes) are used on the data:
        # 1. Put the data into vectors of length 1, average and save only the last one
        # 2. Save all of the raw data directly
        out_str1.buffer(2).average().save("out_avg")
        out_str1.save_all("out_raw")
```
5. Multi-dimensional buffer.

This examples shows usage of a multi-dimensional buffer in the stream processing.
```python
with program() as streamProg_buffer:
    out_str = declare_stream()

    a = declare(int)
    b = declare(int)
    with for_(a, 0, a <= 10, a + 1):
        with for_(b, 10, b < 40, b + 10):
            save(b, out_str)

    with stream_processing():
        out_str.buffer(11, 3).save("out")
```

## Post processing 

N/A

## Script

[download script](intro_to_saving.py)
