---
title: Introduction to streams
sidebar_label: Intro to streams
slug: ./
id: index
---

This example shows basic usage of streams and stream processing. Read more on the stream processing and 
it's capabilities in the [QUA docs](https://qm-docs.qualang.io/guides/stream_proc).

The program consists of a for-loop which populates `stream1` with integers between 0 and 99 and `stream2` with random integers.
So, if you didn't know QUA has a pseudo-random number generator, now you know. 

```python
  with for_(ind,0,ind<100,ind+1):
        save(ind,stream1)
        assign(temp,Random().rand_int(10))
        save(temp,stream2)
```
Once the streams are populated, we can use `stream_processing` to shape and manipulate them in useful ways, 
as specified in the comments associated with each manipulation. Note how you can perform different 
manipulations on the same stream and save the result to different tags ("names"). 

```python

 with stream_processing():
        stream1.save_all('stream1') #saving all samples to a single vector
        stream1.buffer(10).save_all('stream2') #each elements has size 10
        stream1.buffer(10,10).save_all('2d_buffer') #each elements has size 10X10
        stream1.buffer(10).average().save_all('stream2avg') #each elements has size 10 and is averaged column wise. cumulative average returned 
        stream1.buffer(10).average().save('stream2avg_single') #each elements has size 10 and is averaged column wise. only final average returned
        stream1.buffer(3).map(FUNCTIONS.average()).save_all('buffer_average') #Data is first collected to bunches of size 3 and then each bunch is averaged 
        stream2.zip(stream1).save_all('zipped_streams') #two streams are combined into a vector of tuples. like the python zip function.
```

Feel free to explore the results and see that they make sense to you!


[download script](intro-to-streams.py)
