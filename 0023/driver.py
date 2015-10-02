#!/usr/bin/python

import cProfile

def d(n):
    if n in (0, 1):
        return 0
    y, q = int(n ** 0.5), [1]
    m = y + 1
    if y * y == n:
        q.append(y)
        m = y
    for x in range(2, m):
        if n % x == 0:
            q.extend([x, int(n / x)])
    return sum(q)

def method1():
    alist = [n for n in range(28124) if n < d(n)]
    s = [True] * 28124
    for a in alist:
        for b in alist:
            if a + b <= 28123:
                s[a + b] = False
            else:
                break
    return sum(k for k in range(28124) if s[k])

def method2():
    abundants = set(i for i in range(1,28124) if d(i) > i) 
    def abundantsum(i): 
        return any(i-a in abundants for a in abundants) 
    return sum(i for i in range(1,28124) if not abundantsum(i))

if __name__ == '__main__':
    for m in (method1, method2):
        cProfile.run('answer = m()')
        print('Answer = {0}'.format(answer))
