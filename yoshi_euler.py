from collections import Counter
from fractions import gcd, Fraction
from itertools import takewhile, product, islice, count, permutations
from math import factorial
from norvig_sudoku import solve as sudoku_solve
from yoshi_helper import fibonacci, nCr, zeller, isqrt
from yoshi_helper import isprime_table, primes, euler_phi, factorize
import data
import numpy as np

def problem0001(n=1000):
    return sum(x for x in range(n) if x % 3 == 0 or x % 5 == 0)

def problem0002(upto=4000000):
    g = takewhile(lambda x: x < upto, fibonacci())
    return sum(x for x in g if x % 2 == 0)

def problem0003(n=600851475143):
    return max(k for k in factorize(n, primes(10000)))

def problem0004():
    plist = (a * b for a in range(900, 1000) for b in range(a, 1000))
    return max(p for p in plist if str(p) == str(p)[::-1])

def problem0005(upto=20):
    a = 1
    for b in range(2, upto + 1):
        a = a // gcd(a, b) * b
    return a

def problem0006(n=100):
    nums = range(n + 1)
    sum_of_squares = sum(x * x for x in nums)
    square_of_sum = sum(x for x in nums) ** 2
    return square_of_sum - sum_of_squares

def problem0007(n=10001):
    return primes(200000)[n - 1]

def problem0008():
    d = [int(k) for k in data.parse_long_string(data.problem0008)]
    return max(d[k] * d[k + 1] * d[k + 2] * d[k + 3] * d[k + 4]
        for k in range(len(d) - 4))

def problem0009():
    s, m = 1000, 1
    while s % (2 * m) != 0 or 4 * m * m <= 1000 or s <= 2 * m * m:
        m += 1
    n = s // (2 * m) - m
    a, b, c = m * m - n * n, 2 * m * n, m * m + n * n
    return a * b * c

def problem0010(upto=2000000):
    return sum(primes(upto))

def problem0011():
    a = data.parse_grid(data.problem0011)
    w, n = 4, len(a)
    def group_gen():
        for x, y in product(range(n), range(n - w + 1)):
            yield [a[x][y + k] for k in range(w)]    # horizontal
            yield [a[y + k][x] for k in range(w)]    # vertical
        for x, y in product(range(n - w + 1), range(n - w + 1)):
            yield [a[x + k][y + k] for k in range(w)]    # diagonals
            yield [a[x + k][y + w - 1 - k] for k in range(w)]
    return max(v1 * v2 * v3 * v4 for v1, v2, v3, v4 in group_gen())

def problem0012():
    p = primes(1000)
    def num_gen():
        for x in count(4, 2):
            yield x // 2
            yield x + 1
    def num_divisors(t):
        m = 1
        for x in factorize(t, p).values():
            m = m * (1 + x)
        return m
    g = num_gen()
    Da, b, Db = 0, 3, 1
    while Da * Db <= 500:
        a, b = b, next(g)
        Da, Db = Db, num_divisors(b)
    return a * b

def problem0013():
    s = data.parse_number_list(data.problem0013)
    return int(str(sum(s))[:10])

def problem0014(ub=1000000):
    d, nums = {1: 1}, range(ub // 2 + 1, ub, 2)
    for n in nums:
        seq = []
        while n not in d:
            seq.append(n)
            n = 3 * n + 1 if n % 2 else n // 2
        k0 = len(seq) + d[n]
        for k, n in enumerate(seq):
            d[n] = k0 - k
    return max(nums, key=d.get)

def problem0015(n=20):
    return nCr(2 * n, n)

def problem0016():
    return sum(int(x) for x in str(2 ** 1000))

def problem0017():
    ones = 'onetwothreefourfivesixseveneightnine'
    teens = 'teneleventwelvethirfourfifsixseveneighnine' + 'teen' * 7
    tens = 'twentythirtyfortyfiftysixtyseventyeightyninety'
    below100 = len(teens + 10 * tens + 9 * ones)
    return (len('onethousand') + 10 * below100 + 100 * len(ones) +
        100 * 9 * len('hundred') + 99 * 9 * len('and'))

def problem0018():
    t = data.parse_grid(data.problem0018)
    row = t[-1]
    for r in range(len(t) - 2, -1, -1):
        row = [max(row[c], row[c+1]) + t[r][c] for c in range(r + 1)]
    return row[0]

def problem0019():
    return sum(1 for y in range(1901, 2001) for m in range(3, 15)
        if zeller(1, m, y) == 1)

def problem0020():
    return sum(int(x) for x in str(factorial(100)))

def problem0021():
    def d(n):
        return sum(x + n // x
            for x in range(2, isqrt(n) + 1)
            if n % x == 0) + 1
    return sum(x for x in range(2, 10000)
        if x == d(d(x)) and not x == d(x))

def problem0022():
    with open('names.txt') as f:
        lines = f.readlines()    
    namelist = sorted(lines[0].strip('"').split('","'))
    letters = 'ABCDEFGHIJKLMNOPQRSTUVWXYZ'
    d = {a: b for a, b in zip(letters, range(1, len(letters) + 1))}
    return sum((k + 1) * sum(d[c] for c in name)
        for k, name in enumerate(namelist))

def problem0025():
    f = lambda x: x < 10 ** 999
    g = takewhile(f, fibonacci())
    return sum(1 for x in g) + 1

# TODO: Prove that the answer must be a cyclic number
def problem0026(n=1000):
    isprime = isprime_table(n)
    return next(k for k in count(999, -1) if isprime[k] and
        not any(10 ** x % k == 1 for x in range(1, k - 1)))

def problem0027():
    isprime = isprime_table(20000)
    
    # b must be prime
    arange = np.arange(-999, 1000)
    brange = np.array([i for i in range(1000) if isprime[i]])
    a = np.tile(arange, len(brange))
    b = np.repeat(brange, len(arange))
    
    n = 1
    while len(a) > 1:
        v = isprime[n ** 2 + a * n + b]    # isprime > np.in1d >> in
        a, b, n = a[v], b[v], n + 1
    return a[0] * b[0]

def problem0028():
    return sum(4 * x * x + 2 * x + 4 for x in range(2, 1001, 2)) + 1

def problem0029():
    r = range(2, 101)
    return len(set(a ** b for a in r for b in r))

def problem0030():
    d = {str(x): x ** 5 for x in range(10)}
    return sum(x for x in range(10, 6 * 9 ** 5)
        if x == sum(d[c] for c in str(x)))

def problem0031(amount=200):
    coins = (1, 2, 5, 10, 20, 50, 100, 200)
    ways = [1] + [0] * amount
    for c in coins:
        for j in range(c, amount + 1):
            ways[j] += ways[j - c]
    return ways[amount]

def problem0032():
    fs, s = frozenset(str(x) for x in range(1, 10)), set()
    for m1 in range(100):
        for m2 in range(m1, 10000):
            p = m1 * m2
            numstr = ''.join([str(m1), str(m2), str(p)])
            if len(numstr) == 9 and frozenset(numstr) == fs:
                s.add(p)
    return sum(s)

def problem0033():
    r = 1
    for den in range(11, 100):
        for num in range(10, den):
            n1, n2 = divmod(num, 10)
            d1, d2 = divmod(den, 10)
            if (n1 == d1 and num * d2 == den * n2 or
                    n1 == d2 and d2 and num * d1 == den * n2 or
                    n2 == d1 and num * d2 == den * n1 or
                    n2 == d2 and d2 and num * d1 == den * n1):
                r = r * Fraction(num, den)
    return r.denominator

# TODO: Justify 100000 as upper bound instead of 9! * 7
def problem0034():
    f = {str(x): factorial(x) for x in range(10)}
    return sum(n for n in xrange(3, 100000)
        if n == sum(f[k] for k in str(n)))

def problem0035():
    isprime = isprime_table(10 ** 6)
    dvs, c = '024568', 4
    for n in range(10, 10 ** 6):
        if isprime[n] and not any(k in str(n) for k in dvs):
            d = len(str(n))
            rlist = [n // 10 ** k + (n % 10 ** k) * 10 ** (d - k)
                for k in range(1, d)]
            if all(isprime[k] for k in rlist):
                c += 1
    return c

def problem0036():
    def is_palindrome(n):
        return str(n) == str(n)[::-1]
    return sum(n for n in range(1000000)
        if is_palindrome(n) and is_palindrome(bin(n)[2:]))

def problem0037():
    def truncated(p):
        n = len(str(p))
        for k in range(1, n):
            yield p // 10 ** k
            yield p % 10 ** k
    isprime = isprime_table(10 ** 6)
    mprimes = (x for x in count(10) if isprime[x] and
        str(x)[0] in '2357' and str(x)[-1] in '357'
        and not any(k in '45680' for k in str(x)[1:-1]))
    tprimes = []
    while len(tprimes) < 11:
        p = next(mprimes)
        if all(isprime[x] for x in truncated(p)):
            tprimes.append(p)
    return sum(tprimes)

def problem0038():
    pandigits = sorted('123456789')
    def find_pandigital_products():
        for n in range(2, 10):
            m, numstr = 0, ''
            while len(numstr) <= 9:
                numstr = ''.join(str(k * m) for k in range(1, n + 1))
                if len(numstr) == 9 and sorted(numstr) == pandigits:
                    yield int(numstr)
                m = m + 1
    return max(find_pandigital_products())

def problem0039():
    s, m_max = set(), int((-1 + 2001 ** 0.5) / 2)
    for m in range(2, m_max):
        for n in range(1, m):
            a, b, c = m * m - n * n, 2 * m * n, m * m + n * n
            g = gcd(gcd(a, b), c)
            a, b, c = a // g, b // g, c // g
            if a + b + c <= 1000:
                s.add(a + b + c)
    c_max, p_max = 0, min(s)
    for p in range(min(s), 1001):
        c = 0
        for k in s:
            if p % k == 0:
                c += 1
                if c > c_max:
                    c_max, p_max = c, p
    return p_max

def problem0040():
    s = ''.join(str(k) for k in range(200000))
    p = 1
    for k in range(7):
        p *= int(s[10 ** k])
    return p

def problem0041():
    isprime = isprime_table(10 ** 7)
    def perms_to_n():
        for n in (7, 4):
            s = ''.join(map(str, range(n, 0, -1)))
            for x in permutations(s, n):
                yield int(''.join(x))
    return next(x for x in perms_to_n() if isprime[x])

def problem0045():
    def p(n):
        return n * (3 * n - 1) // 2
    def h(n):
        return n * (2 * n - 1)
    j, k = 165, 144
    while not h(k) == p(j):
        k += 1
        while p(j) < h(k):
            j += 1
    return h(k)

def problem0046():
    isprime = isprime_table(10000)
    RHS = 2 * np.arange(len(isprime)) ** 2
    x, flag = 7, True
    while flag:
        x = x + 2
        if isprime[x]:
            x = x + 2
        flag = isprime[x - RHS[RHS < x]].any()
    return x

def problem0047():
    fc = np.zeros(1000000, dtype=int)
    for p in primes(int(len(fc) ** 0.5 + 1)):
        fc[p::p] += 1
    fours = np.flatnonzero(fc == 4)
    return next(x for x in fours
        if fc[x + 1] == fc[x + 2] == fc[x + 3] == 4)

def problem0048():
    return sum(x ** x for x in range(1,1001)) % (10 ** 10)

def problem0049():
    isprime = isprime_table(10000)
    primefilter = lambda p: isprime[p] and not p == 4817
    def find_triplet():
        for p in filter(primefilter, range(1000, 10000)):
            perms = permutations(str(p))
            v = np.array([int(''.join(x)) for x in perms])
            d = np.unique(v[isprime[v]]) - p
            delta = [x for x in d
                if x > 0 and -x in d and p - x > 1000]
            if delta:
                yield p - delta[0], p, p + delta[0]
    return int('{}{}{}'.format(*next(find_triplet())))

def problem0050(upto=1000000):
    isprime = isprime_table(upto)
    p = np.flatnonzero(isprime)
    num_terms, found_prime = 546, None
    while not found_prime:
        i, psum = 0, p[:num_terms].sum()
        while psum < upto and not found_prime:
            found_prime = isprime[psum]
            if not found_prime:
                i = i + 1
                psum = p[i:i+num_terms].sum()
        num_terms = num_terms - 1
    return psum

def problem0052():
    ss = lambda x: sorted(str(x))
    return next(x for x in count(1) if
        ss(x) == ss(2 * x) == ss(3 * x) ==
        ss(4 * x) == ss(5 * x) == ss(6 * x))

# TODO: complete solution
def problem0054():
    cardranks = {c: k for k, c in enumerate('23456789TJQKA')}
    cardranksAceLow = {c: k for k, c in enumerate('A23456789TJQK')}
    def rankHand(h):
        ranks = Counter(c[0] for c in h)
        rankgroups = Counter(ranks.values())
        suits = Counter(c[1] for c in h)
        isflush = len(suits) == 1
        straightranks = ([cardranksAceLow[c] for c in ranks]
            if 5 in ranks else [cardranks[c] for c in ranks])
        isstraight = (max(straightranks) - min(straightranks) == 5 and
            1 in rankgroups and rankgroups[1] == 5)
        return next(n for n, e in enumerate([
            isflush and isstraight,                   # straight flush
            4 in rankgroups,                          # four of a kind
            3 in rankgroups and 2 in rankgroups,      # full house
            isflush,                                  # flush
            isstraight,                               # straight
            3 in rankgroups,                          # three of a kind
            2 in rankgroups and rankgroups[2] == 2,   # two pair
            2 in rankgroups,                          # pair
            True]) if e)
    def tieNothing(h1, h2):
        ranks1 = sorted(cardranks[c[0]] for c in h1)
        ranks2 = sorted(cardranks[c[0]] for c in h2)
        return ranks1[::-1] > ranks2[::-1]
    def tiePair(h1, h2):
        ranks1, ranks2 = [c[0] for c in h1], [c[0] for c in h2]
        return tieNothing([x for x in h1], [x for x in h2])
    c = 0
    for h1, h2 in data.readpoker():
        r1, r2 = rankHand(h1), rankHand(h2)
        if r1 < r2:
            c = c + 1
        elif r1 == r2 == 8 and tieNothing(h1, h2):
            c = c + 1
        elif r1 == r2 == 7 and tiePair(h1, h2):
            c = c + 1
    return 42

def problem0057():
    def sqrt2fractions():
        n, d = 2, 1
        while True:
            yield n + d, n
            n, d = 2 * n + d, n
    return sum(1 for n, d in islice(sqrt2fractions(), 1000)
        if len(str(n)) > len(str(d)))

def problem0059():
    cipher = data.readCipher1()
    kcombos = product(range(ord('a'), ord('z') + 1), repeat=3)
    attempts = (''.join(chr(cipher[k] ^ kchars[k % 3])
                for k in range(len(cipher)))
            for kchars in kcombos)
    cracked = next(a for a in attempts if a.count('the') > 5)
    return sum(ord(c) for c in cracked)

def problem0065():
    def ecfdeccoefficients():
        yield 2
        for k in count(1):
            yield 1; yield 2 * k; yield 1
    coeff = list(islice(ecfdeccoefficients(), 100))
    k, f = len(coeff) - 1, Fraction(1, coeff[-1])
    for c in reversed(coeff[1:-1]):
        f = 1 / (c + f)
    f = coeff[0] + f
    return sum(int(x) for x in str(f.numerator))

def problem0067():
    with open('triangle.txt') as f:
        lines = f.readlines()
    t = data.parse_grid(''.join(lines))
    row = t[-1]
    for r in range(len(t) - 2, -1, -1):
        row = [max(row[c], row[c+1]) + t[r][c] for c in range(r + 1)]
    return row[0]

def problem0078():
    maxK = 500
    klist = np.r_['0,2', 1:maxK, -1:-maxK:-1].ravel(order='F')
    slist = np.ones(klist.shape, dtype='int64')
    slist[2::4] = -1; slist[3::4] = -1
    gp = klist * (3 * klist - 1) // 2

    p = np.r_[1, 1, np.zeros(100000, dtype='int64')]
    n, maxI = 1, 0
    while p[n] % 1000000:
        n = n + 1
        while gp[maxI] <= n:
            maxI = maxI + 1
        p[n] = (slist[:maxI] * p[n - gp[:maxI]]).sum() % 1000000
    return n

def problem0079():
    def check_rule(rule, key):
        k, c = 0, 0
        while c < 3 and k < len(key):
            c = c + (key[k] == rule[c])
            k = k + 1
        return c == 3
    rules = data.readkeylog()
    nset = ''.join(set(''.join(rules)))
    v = next(key for key in permutations(nset)
        if all(check_rule(r, key) for r in rules))
    return int(''.join(v))

def problem0089():
    def char_saved(s):
        v = s.replace('DCCCC', 'CM')
        v = v.replace('CCCC', 'CD')
        v = v.replace('LXXXX', 'XC')
        v = v.replace('XXXX', 'XL')
        v = v.replace('VIIII', 'IX')
        v = v.replace('IIII', 'IV')
        return len(s) - len(v)
    return sum(char_saved(r) for r in data.readRoman())

def problem0096():
    def puzzles(filename):
        with open(filename) as f:
            lines = f.read().strip().split('\n')
        for k in range(50):
            yield ''.join(lines[1+10*k:10+10*k])
    def topLeftThree(s):
        a, b, c = s['A1'], s['A2'], s['A3']
        return int(a + b + c)
    return sum(topLeftThree(sudoku_solve(s))
        for s in puzzles('sudoku.txt'))

def problem0092(upto=10000000):
    def ends89(n):
        while n not in (0, 1, 89):
            n = sum(int(x) ** 2 for x in str(n))
        return n == 89
    ndigits = len(str(upto)) - 1
    sq = np.array([x * x for x in range(10)], dtype='int32')
    f = np.zeros((ndigits * sq[-1] + 1, ndigits + 1), dtype='int32')
    f[0, 0] = 1
    for k in range(1, f.shape[1]):
        for n in range(f.shape[0]):
            v = n - sq
            f[n, k] = f[v[v >= 0], k - 1].sum()
    mask = np.array([ends89(x) for x in range(f.shape[0])])
    return sum(f[:, -1] * mask)

def problem0097(digits=10):
    m = 10 ** digits
    return (28433 * pow(2, 7830457, m) + 1) % m

def problem0243():
    cutoff = Fraction(15499, 94744)
    p = primes(100)
    def resilience(n):
        return euler_phi(n, p) * Fraction(1, n - 1)

    x, k = p[0], 1
    while resilience(x * p[k]) >= cutoff:
        x, k = x * p[k], k + 1
    k = 2
    while resilience(x * k) >= cutoff:
        k = k + 1
    return x * k

# TODO: Find code for problem 53, 55, 56, 73, 81, 99
# TODO: Transfer code for problem: 23-24, 42-44
