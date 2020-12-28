---
title: Intro to macros
sidebar_label: intro-to-macros
slug: ./
id: index
---

It is possible to use python functions inside QUA programs. This technique where one 
programming language is used to write another is called _metaprogramming_ and is a great way 
to write more powerful and readable QUA code. 

The program defined in `intro-to-macros.py` showcase basic usage of macros. 
A declaration of variables is first done in a macro. In this case, the variables must 
be explicitly returned to the main program for them to be in its scope. 
This program also returns an array of output streams, with default size =1. 

In experiments where preparation and processing stages are needed, for example in quantum-dots 
and cold-atom experiments, separating those into macros are recommended. Such steps can be parameterized
as demonstrated in this example file.

## Script

[download script](intro-to-macros.py)
