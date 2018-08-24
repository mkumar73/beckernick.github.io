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
# 1 1 2 3 5 8 13 21 .......

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
Let's see the result and the time taken for them to completed for two different series one
when `n=10` and other with `n=40`.

Results:
```python
# time representation : h:mm:ss:ns
# for n = 10: 55
# time taken for n = 10: 0:00:00.000049
# for n = 40: 102334155
# time taken for n = 40: 0:00:39.934757
```

## Drawbacks of Recursion

If you notice carefully, the time taken for computing the 40th element of a Fibonacci's series
is not at all linear if we compare it with `n=10`. The reason behind it is that in order to
calculate the element of longer fibonacci series the loop has to go back again and again for
each iteration without maintaining any hash table to store the previously calculated result.

**Time complexity : O(2^n)**
  
For illustration, lets see the steps involved for calculating `fib(4)`.

1. 1st and 2nd element will be returned as 0 and 1.
  So, fib(1) and fib(2) will 0 and 1 respectively.
2. For 3rd element function has to perform `fib(3-1) + fib(3-2)`.
  and now, fib(3) = fib(2) + fib(1) = 1 + 0 = 1
3. Now for 4th element function has to perform `fib(4-1) + fib(4-2)`.
   
   a. fib(4) = fib(3) + fib(2)
   
   b. the value to fib(3) and fib(2) are not stored any where, so it has to start
      again from step 2.

As a result, for larger series, it is not advisable to use recursion as it might get
quite time intensive during run-time. However, one can always improve the implementation 
if possible depending on the use case.


## Dynamic Programming and Memoization

It has been observed that most of the people get confused between Dynamic programming and
Memoization. So, let's define it one by one.

**Dynamic Programming** is a technique for solving problems recursively and is applicable when the 
computations of the sub-problems have same implementation or overlap.

**Memoization** is a technique used in computing to speed up programs. This is accomplished by 
memorizing the calculation results of processed input such as the results of function calls. 
If the same input or a function call with the same parameters is used, the previously stored 
results can be used again and unnecessary calculation are avoided. It is a "Top-down" approach
in a sense that you solve the top problem first which in turn uses recursion to solve the sub-problems.

Dynamic programming is typically implemented using tabulation, but can also be implemented 
using memoization. It is basically implemented in "Bottom-up" fashion i.e. solving all related 
sub-problems first and filling up the table and then use the result from the table to compute the
solution of the original problem.

But there are some important points which should be considered before implementing either of them:
1. If all sub-problems must be solved at least once, a bottom-up dynamic-programming 
algorithm usually outperforms a top-down memoization algorithm.
2. If some sub-problems in the problem space need not be solved at all, then the memoization
 technique can be well suited as it doesn't evaluate the unnecessary sub-problem.
 

## Solution to Fibonacci using Memoization (Top-down)

Let's see the implementation of the Fibonacci's series using `Memoization technique`.


```python
# using memoization

from datetime import datetime

def fib_memoization(n, d):

    if n in d.keys():
        return d[n]
    else:
        result = fib_memoization(n-1, d) + fib_memoization(n-2, d)
        d[n] = result
        return result


# dictionary for the function fib_memoization
d = {1: 1, 2: 1}

# time taken for n=10
start = datetime.now()
print('for n = 10:', fib_memoization(10, d))
end = datetime.now()
print('time taken for n = 10:', end - start)

# time taken for n=40
start = datetime.now()
print('for n = 40:', fib_memoization(40, d))
end = datetime.now()
print('time taken for n = 40:', end - start)
```

Results:
```python
# for n = 10: 55
# time taken for n = 10: 0:00:00.000011
# for n = 40: 102334155
# time taken for n = 40: 0:00:00.000020
```

*Observations:* As compared to normal recursion implementation, it is a huge improvement with 
respect to time complexity. The time taken for finding the 10th element of the series is almost 
similar for both the approach but memoization shines for `n=40`. The function `fib_memoization`
is passed with two parameters, first the value of `n` and a dictionary for storing the intermediate
values. 

**Time complexity : O(n)**

During execution, the dictionary keeps track of all the values computed and stores them as a key-value
pair and it goes to the loop only when the key is not present in the map or hash table and evaluate the results. 
As a result, for `n=40` it has to evaluate the loop far less times as compared with the original 
recursion example.

## Solution to Fibonacci using Dynamic Programming (Bottom-up)

Let's see the implementation of the Fibonacci's series using `Bottom up approach`.


```python
# using dynamic programming (bottom-up approach)

import datetime

def fib_bottom_up(n):

    bottom_up = dict()
    bottom_up[1] = 1
    bottom_up[2] = 1

    if n == 1 or n == 2:
        return 1

    for i in range(3, n+1):
        bottom_up[i] = bottom_up[i-1] + bottom_up[i-2]

    return bottom_up[n]


# time taken for n=10
start = datetime.now()
print('for n = 10:', fib_bottom_up(10))
end = datetime.now()
print('time taken for n = 10:', end - start)

# time taken for n=40
start = datetime.now()
print('for n = 40:', fib_bottom_up(40))
end = datetime.now()
print('time taken for n = 40:', end - start)
```

Results:
```python
# for n = 10: 55
# time taken for n = 10: 0:00:00.000030
# for n = 40: 102334155
# time taken for n = 40: 0:00:00.000050
```

*Observations:* The time taken to evaluate the 10th element is quite comparable for all the three 
approaches but we can easily see that the result for larger series we should always consider using
either the `Dynamic Programming` or `Memoization` approach.

**Time complexity : O(n)**

In case of Bottom-up approach as well, the values are stored in the hash table, in this case the dictionary,
but the loop has to be evaluated all the time and restore the result. In contrary to Memoization where 
it enter the loop only if the required key is not present in the hash map.

`Note:` The time taken to complete the process depends on the system we use, so it will definitely vary
for your system. 

## Class based implementation

In real scenarios, it is always advisable to use the Object Oriented Programming (OOP) construct 
whenever possible for code maintainability and clarity. Keeping it in mind, we have added the solution.
See the below implementation for your reference.

```python
class Fibonacci:

    def __init__(self):
        # store the result for future use
        self.cache = {}

    def __call__(self, n):
        if n not in self.cache:
            if n == 1:
                self.cache[1] = 1
            elif n == 2:
                self.cache[2] = 1
            else:
                self.cache[n] = self.__call__(n-1) + self.__call__(n-2)
        return self.cache[n]


fib = Fibonacci()

for i in range(1, 11):
    print(fib(i), end=", ")

# Result:
# Using python class:
# 1, 1, 2, 3, 5, 8, 13, 21, 34, 55,
```

## Class and decorator based implementation

Decorators are one of the important topic in python programming and we will discuss it
in details in some other post but for now, just follow it's application for the current
use case.

```python
# class and decorator based memoization

class Memoize:

    def __init__(self, fn):
        self.fn = fn
        self.memo = {}

    def __call__(self, *args):
        if args not in self.memo.keys():
            self.memo[args] = self.fn(*args)
        return self.memo[args]


@Memoize
def fib(n):
    if n == 1 or n == 2:
        return 1
    else:
        return fib(n-1) + fib(n-2)


print('Using Class based approach:')
for i in range(1, 11):
    print(fib(i), end=', ')
    
# Result:
# Using python class and decorators::
# 1, 1, 2, 3, 5, 8, 13, 21, 34, 55,
```


## Conclusion

After, verifying the results for Fibonacci's series, we can say that the normal recursion approach is 
not suitable for complicated scenarios. This is just one of the use case that I tried to cover but
in reality, the situation can be more challenging. So, the solutions also depends on the kind of 
problem.

Disclaimer: The example presented is just for the purpose of demonstration, there can always be
a better approach for the same use case. Also, please comment below in case you want to see some
specific posts like this. Please see the
[Github Repository](https://github.com/mkumar73/py_practice/blob/master/recursion_memoization_dp.py) 
 for the complete code.Thank you.:)
  

