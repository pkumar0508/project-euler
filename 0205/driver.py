#!/usr/bin/python

import cProfile

def getProbabilities(num_dice, num_faces):
    outcomes = range(num_dice, num_faces * num_dice + 1)
    p, q = 1.0 / num_faces, 1.0 - 1.0 / num_faces
    bin_dist = [p ** num_dice]
    for k in range(num_dice * (num_faces - 1) + 1):
        old_pmf_val = bin_dist[-1]
        bin_dist.append()

def method1():
    return 42

if __name__ == '__main__':
    for m in (method1,):
        cProfile.run('answer = m()')
        print('Answer = {0}'.format(answer))
