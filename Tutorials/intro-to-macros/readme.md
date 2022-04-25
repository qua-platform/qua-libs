---
title: Intro to macros
sidebar_label: Intro to macros
slug: ./
id: index
---

It is possible to use python functions inside QUA programs. This technique where one 
programming language is used to write another is called _metaprogramming_ and is a great way 
to write more powerful and readable QUA code. 

In experiments where preparation and processing stages are needed, for example in quantum-dots 
and cold-atom experiments, separating those into macros are recommended. Such steps can be parameterized
as demonstrated in this example file.

The program defined in `intro-to-macros.py` showcase basic usage of macros. 
A declaration of variables is first done in a macro. In this case, the variables **must 
be explicitly returned** to the main program for them to be in its scope. 
This program also returns an array of output streams, with default size =1. 

```python
def declare_vars(stream_num=1):
    """
    A macro to declare QUA variables. Stream num showcases a way to declare multiple streams in an array
    Note that variables and streams need to be explicitly returned to the QUA function to be in scope
    """
    time_var = declare(int, value=100)
    amp_var = declare(fixed, value=0.2)
    stream_array = [declare_stream() for num in range(stream_num)]
    return [time_var, amp_var, stream_array]

```
Then we demonstrate how QUA variables can be modified within a macro 


The program then demonstrates how QUA statements can be initiated with a macro and parameterized 
such that reusable components can be set up. 

```python
def qua_function_calls(el):
    """
    A macro that calls QUA play statements
    :param el: The quantum element used by the QUA statements
    :return:
    """
    play('const', el, duration=300)
    play('const'*amp(b), el, duration=300)
```
## Script

[download script](intro-to-macros.py)
