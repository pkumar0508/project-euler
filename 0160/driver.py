#!/usr/bin/python

import cProfile
import math

max_num = 10 ** 6

def strip_trailing_zeros(n):
    while n % 10 == 0:
        n = int(n / 10)
    return n

def method1():
    product = 1
    num_digits = 5
    mod_factor = 10 ** num_digits
    for x in range(2, max_num + 1):
        product *= strip_trailing_zeros(x) % mod_factor
        product = strip_trailing_zeros(product) % mod_factor
    return product

def method2():
    return int(str(math.factorial(max_num)).rstrip('0')[-5:])
    
if __name__ == '__main__':
    #for m in (method1, method2):
    for m in (method1,):
        cProfile.run('answer = m()')
        print('Answer = {0}'.format(answer))
