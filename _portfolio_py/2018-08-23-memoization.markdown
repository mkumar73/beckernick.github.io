---
title:  "Dynamic Programming and Memoization Demystified"
date:   2018-08-24
tags: [python, advance]

excerpt: "recursion, dynamic programming, memoization"
---


## Recursion and its basic implementation

It is fairly important to know *Recursion* before talking about *Dynamic Programming*.
So, what is **Recursion**?  As per definition, Recursion is the process of repetition 
of a sequence of computer instruction or a command specified number of times or 
until the condition is met.

Recursion is quite a common programming paradigm and we all have learned it 
during our studies. The first thing that comes to our mind (at least for me) when we 
talk about recursion is the popular **Tower of Hanoi** problem and **Fibonacci's series**
and their implementation using recursion.

So, for this post, we will implement Fibonacci series using recursion, and then 
we will see, how it can be improved using Dynamic Programming paradigm and Memoization.

```python

# fibonacci series example:
# 0 1 1 2 3 5 8 13 21 .......

# program below returns the nth element of the series.

# fib using recursion
from datetime import datetime

def fib(n):
    if n == 0:
        return n
    elif n == 1:
        return n
    else:
        return fib(n - 1) + fib(n - 2)


# time taken for n=10
start = datetime.now()
print('for n = 10:', fib(10))
end = datetime.now()
print('time taken for n = 10:', end - start)

# time taken for n=40
start = datetime.now()
print('for n = 40:', fib(40))
end = datetime.now()
print('time taken for n = 40:', end - start)
```

The *else* part of the function *fib(n)* implements recursion as it calls the function
itself with the parameter of the last and the second last number of the fibonacci series.
Let' see the result and time taken for them to completed for two different series one
when 'n=10' and other with 'n=40'.

Results:
```python
# time representation : h:mm:ss:ns
# for n = 10: 55
# time taken for n = 10: 0:00:00.000049
# for n = 40: 102334155
# time taken for n = 40: 0:00:39.934757
```

## Drawbacks of Recursion

If you notice carefully, the time taken for computing 40th element of a Fibonacci's series
is not at all linear if we compare it with `n=10`. The reason behind it is that in order to
calculate the element of longer fibonacci series the loop has to go back again and again for
every iteration without maintaining any hash table to store the previously calculated result.

Time complexity of recursion is 
  
For illustration, lets see the steps involved for calculating `fib(4)`.

1. 1st and 2nd element will be returned as 0 and 1.
  So, fib(1) and fib(2) will 0 and 1 respectively.
2. For 3rd element function has to perform `fib(3-1) + fib(3-2)`.
  and now, fib(3) = fib(2) + fib(1) = 1 + 0 = 1
3. Now for 4th element function has to perform `fib(4-1) + fib(4-2)`.
   
   a. fib(4) = fib(3) + fib(2)
   
   b. the value to fib(3) and fib(2) are not stored any where, so it has to start
      again from step 2.

As a result, for larger series, it is not advisable to use recursion as it is and improve
the implementation if possible.


## Dynamic Programming and Memoization

It has been observed that most of the people get confused between Dynamic programming and
Memoization. So, it define it one by one.

**Dynamic Prgramming**

