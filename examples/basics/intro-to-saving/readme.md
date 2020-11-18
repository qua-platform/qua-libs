---
title: Intro to saving
sidebar_label: Intro to saving
slug: ./
id: index
---

This example shows four QUA programs where variables are saved.
Each one shows a slightly different variation on how this can be achieved:
either by saving directly to a tag that is then collected from the `result_handles`
structure, or by saving to a `stream` and processing in various ways. 

## Config

The configuration for this example is included, but irrelevant as no pulses are 
played to any output and no data is read in. 

## Programs

Four programs are included: 
1. Assigning values variables and saving variables to tags.
This program saves literal values and values calculated with math operations. 
It then saves them directly to `tags`. This is a less powerful method of saving
that is not the recommended mode of operation, yet it is still supported. 

2. Saving variables to streams and using stream processing.
The `stream` construct is a powerful way to save and manipulate data on the server. 
It is described in detail in the QUA main docs. This example shows basic usage. 

3. Using the buffer operator in stream processing.

Streams can be acted upon with various operators. The buffer operator is a 
particularly useful one which allows to reshape incoming data into buffers of 
a predefined size. Using additional operators, such as the `average()` operator 
as shown here, allows averaging such buffers with one another. In this case 
all average buffers are saved, such that you can track the evolution of the average.  

4. Saving a stream to multiple tags.

This examples shows how multiple output variables can be defined in one `steam_processing`
environment. 


## Post processing 

N/A

## Script

[download script](intro_to_saving.py)
