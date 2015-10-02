#!/usr/bin/python

def modular_helper(base, exponent, modulus, prefactor=1):
    c = 1
    for k in range(exponent):
        c = (c * base) % modulus
    return ((prefactor % modulus) * c) % modulus

def fibN(n):
    phi = (1 + 5 ** 0.5) / 2
    return int(phi ** n / 5 ** 0.5 + 0.5)

# Alternate problem solutions start here

def problem0012a():
    p = primes(1000)
    n, Dn, cnt = 3, 2, 0
    while cnt <= 500:
        n, n1 = n + 1, n
        if n1 % 2 == 0:
            n1 = n1 // 2
        Dn1 = 1
        for pi in p:
            if pi * pi > n1:
                Dn1 = 2 * Dn1
                break
            exponent = 1
            while n1 % pi == 0:
                exponent += 1
                n1 = n1 / pi
            if exponent > 1:
                Dn1 = Dn1 * exponent
            if n1 == 1:
                break
        cnt = Dn * Dn1
        Dn = Dn1
    return (n - 1) * (n - 2) // 2

def problem0013a():
    with open('problem0013.txt') as f:
        s = f.readlines()
    return int(str(sum(int(k[:11]) for k in s))[:10])

# solution due to veritas on Project Euler Forums
def problem0014a(ub=1000000):
    table = {1: 1}
    def collatz(n):
        if not n in table:
            if n % 2 == 0:
                table[n] = collatz(n // 2) + 1
            elif n % 4 == 1:
                table[n] = collatz((3 * n + 1) // 4) + 3
            else:
                table[n] = collatz((3 * n + 1) // 2) + 2
        return table[n]
    return max(xrange(ub // 2 + 1, ub, 2), key=collatz)

# 13 -> 40 -> 20 -> 10 -> 5 -> 16 -> 8 -> 4 -> 2 -> 1
# 13 -(3)-> 10 -(1)-> 5 -(3)-> 4 -(1)-> 2 -(1)-> 1
def veritas_iterative(ub=1000000):
    table = {1: 1}
    def collatz(n):
        seq, steps = [], []
        while not n in table:
            seq.append(n)
            if n % 2 and n % 4 == 1:
                n, x = (3 * n + 1) // 4, 3
            elif n % 2:
                n, x = (3 * n + 1) // 2, 2
            else:
                n, x = n // 2, 1
            steps.append(x)
        x = table[n]
        while seq:
            n, xn = seq.pop(), steps.pop()
            x = x + xn
            table[n] = x
        return x
    return max(xrange(ub // 2 + 1, ub, 2), key=collatz)

def problem0026a(n=1000):
    return max(d for d in primes(n)
        if not any(10 ** x % d == 1 for x in range(1, d - 1)))

def problem0031a():
    def tally(*p):
        d = (100, 50, 20, 10, 5, 2, 1)
        return 200 - sum(k * v for k, v in zip(p, d))
    c = 2
    for p100 in range(2):
        mp50 = int(tally(p100) / 50) + 1
        for p50 in range(mp50):
            mp20 = int(tally(p100, p50) / 20) + 1
            for p20 in range(mp20):
                mp10 = int(tally(p100, p50, p20) / 10) + 1
                for p10 in range(mp10):
                    mp5 = int(tally(p100, p50, p20, p10) / 5) + 1
                    for p5 in range(mp5):
                        mp2 = int(tally(p100, p50, p20, p10, p5) / 2) + 1
                        for p2 in range(mp2):
                            c += 1
    return c

def problem0089a():
    n2r = [(1000, 'M'), (900, 'CM'), (500, 'D'), (400, 'CD'),
        (100, 'C'), (90, 'XC'), (50, 'L'), (40, 'XL'), 
        (10, 'X'), (9, 'IX'), (5, 'V'), (4, 'IV'), (1, 'I')]
    r2n = {b: a for a, b in n2r}
    def to_roman(x):
        s = []
        while x:
            n, c = next((n, c) for n, c in n2r if x >= n)
            s.append(c)
            x = x - n
        return ''.join(s)
    def from_roman(r):
        k, s = 0, 0
        while k < len(r):
            if r[k] not in ('I', 'X', 'C') or k == len(r) - 1:
                s = s + r2n[r[k]]
            elif r[k:k+2] in r2n:
                s = s + r2n[r[k:k+2]]
                k = k + 1
            else:
                s = s + r2n[r[k]]
            k = k + 1
        return s
    return sum(len(r) - len(to_roman(from_roman(r)))
        for r in data.readRoman())

def problem0097a():
    # Note 7830457 = 29 * 270015 + 22
    # (10 ** 10 - 1) * 2 ** 29 does not overflow a 64 bit integer
    p, b, e = 28433, 2, 7830457
    d, m = divmod(e, 29)
    prefactor = 28433 * 2 ** m
    return modular_helper(2 ** 29, 270015, 10 ** 10, 28433 * 2 ** m) + 1
