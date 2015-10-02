#!/usr/bin/python

import time

def tn_gen(upto):
    k, n = 1, 1
    while n <= upto:
        yield n
        k, n = k + 1, n + k + 1

letters, d = 'ABCDEFGHIJKLMNOPQRSTUVWXYZ', {}
for a, b in zip(letters, range(1, len(letters) + 1)):
    d[a] = b

def score(word):
    return sum(d[c] for c in word)

def method1():
    with open('words.txt') as f:
        lines = f.readlines()    
    wordlist = sorted(lines[0].strip('"').split('","'))
    count = 0
    q = [score(word) for word in wordlist]
    m = max(q)
    tn = [x for x in tn_gen(m)]
    for c in q:
        if c in tn:
            count += 1
    return count

if __name__ == '__main__':
    start_time = time.time()
    a = method1()
    end_time = time.time()
    print('Answer: {0}'.format(a))
    print('Time elapsed: {0}'.format(end_time - start_time))
