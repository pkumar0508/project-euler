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
    yield int(''.join(str(d) for d in a))
    j, k = find_indices(a)
    while not j == -1:
        a[j], a[k] = a[k], a[j]
        a[j+1:] = a[j+1:][::-1]
        yield int(''.join(str(d) for d in a))
        j, k = find_indices(a)

def ss(n, k):
    return int(''.join([d for d in str(n)[k-1:k+2]]))

def method1():
    g = lex_gen()
    return sum(n for n in g
               if ss(n, 2) % 2 == 0 and
                  ss(n, 3) % 3 == 0 and
                  ss(n, 4) % 5 == 0 and
                  ss(n, 5) % 7 == 0 and
                  ss(n, 6) % 11 == 0 and
                  ss(n, 7) % 13 == 0 and
                  ss(n, 8) % 17 == 0)

if __name__ == '__main__':
    for m in (method1,):
        cProfile.run('answer = m()')
        print('Answer = {0}'.format(answer))
