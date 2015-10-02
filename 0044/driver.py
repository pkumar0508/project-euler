#!/usr/bin/python

import cProfile
import numpy as np
import itertools

def p(n):
    p, x = 1, 1
    while p < n:
        yield p
        x = x + 1
        p = int(x * (3 * x - 1) / 2)

def gen_ispentagonal():
    n, p = 1, 1
    for x in itertools.count():
        yield x == p
        if x == p:
            n = n + 1
            p = int(n * (3 * n - 1) / 2)

def method1():
    max_val = 10 ** 8
    ispentagonal = np.zeros(max_val, dtype=bool)
    ispentagonal[np.array([x for x in p(max_val)])] = True
    pentagonal = np.array([x for x in p(max_val / 2)])

    a = np.tile(pentagonal, len(pentagonal))
    b = np.repeat(pentagonal, len(pentagonal))
    
    v = a > b
    a, b = a[v], b[v]
    add, sub = a + b, a - b
    v = np.logical_and(ispentagonal[add], ispentagonal[sub])

    return sub[np.flatnonzero(v)][0]

if __name__ == '__main__':
    for m in (method1,):
        cProfile.run('answer = m()')
        print('Answer = {0}'.format(answer))
