#!/usr/bin/python

import cProfile

def lex_gen():
    def find_indices(a):
        j, k = len(a) - 2, -1
        while not a[j] < a[j + 1]:
            j -= 1
            if j == -1:
                return (j, k)
        k = len(a) - 1
        while not a[j] < a[k]:
            k -= 1
        return (j, k)
    
    a = range(10)
    yield a
    j, k = find_indices(a)
    while not j == -1:
        a[j], a[k] = a[k], a[j]
        a[j+1:] = a[j+1:][::-1]
        yield a
        j, k = find_indices(a)

def method1():
    g = lex_gen()
    for x in range(1000000):
        n = g.next()
    return ''.join(str(x) for x in n)

def method2():
    f = [1]
    for x in range(1, 11):
        f.append(x * f[-1])
    r, q, n = range(10), [], 1000000 - 1
    while r:
        k = int(n / f[len(r) - 1])
        n -= k * f[len(r)  - 1]
        q.append(r.pop(k))
    return ''.join(str(x) for x in q)

if __name__ == '__main__':
    for m in (method1,method2):
        cProfile.run('answer = m()')
        print('Answer = {0}'.format(answer))
